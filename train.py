import argparse
import copy
import json
import os
import random
import time
from collections import defaultdict
from datetime import datetime
from ntpath import basename

# import aim
import cv2
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from torch import hub
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

from dataloaders.ct_lungs import LungsDataset
from metrics import calc_f1, calc_loss, print_metrics
from policy import (RandomLearner, UncertainLearnerEntropy,
                    VarianceLearnerDropout)
from resumable_sampler import ResumableRandomSampler


def plot_grad_flow(epoch, idx, named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        print(n, p.requires_grad, p.is_leaf, p.grad is not None)
        if p.requires_grad and p.is_leaf and ("bias" not in n):
            layers.append(n[: n.rfind('.')])
            ave_grads.append(p.grad.abs().mean().cpu().numpy() if p.grad is not None else 0.)
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.savefig(f'tmp/trash/plots_{epoch}_{idx}.png')


def train_model(model, dataloaders, policy_learner, optimizer, scheduler, num_epochs, device, writer_name):
    writer = SummaryWriter(writer_name, flush_secs=1)
    # aim.set_params({'dataset': writer_name, 'policy': class_name}, name='config')
    loader = {'val': dataloaders['val']}

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    n_images = {'train': 0, 'val': 0}

    for epoch in range(num_epochs):
        loader['train'] = policy_learner()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('+++++++++ len loader', len(loader[phase]))
            if phase == 'train':
                if scheduler:
                    scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for enum_id, (idxs, inputs, labels) in tqdm(enumerate(loader[phase]), total=len(loader[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)
                # if phase == 'train' and enum_id < 3:
                #     for idx in idxs:
                #         torch.save(torch.tensor(1),
                #                    f'tmp/trash/{policy_learner.__class__.__name__}_{epoch}_{enum_id}__{idx}'
                #                    )

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss, loss_sum, loss_bce, loss_dice = calc_loss(outputs, labels)
                    acc_f1 = calc_f1(outputs, labels)
                    # acc_iou = calc_IOU(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        # plot_grad_flow(epoch, enum_id, model.named_parameters())
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                n_images[phase] += inputs.size(0)

                writer.add_scalar(f'{phase}/loss', loss_sum, n_images[phase])
                # aim.track(loss_sum.item(), name='loss', epoch=n_images[phase], phase=phase)
                # writer.add_scalar(f'{phase}/bce', loss_bce, n_images[phase])
                # writer.add_scalar(f'{phase}/dice', loss_dice, n_images[phase])

                # aim.track(loss_sum, f'{phase}/loss', n_images[phase])

                metrics['loss'] += loss * inputs.size(0)
                metrics['f1'] += acc_f1 * inputs.size(0)
                # metrics['iou'] += acc_iou * inputs.size(0)

            print_metrics(writer, metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            writer.add_scalar(f'{phase}/epoch_loss', epoch_loss, epoch)
            epoch_f1 = metrics['f1'] / epoch_samples
            writer.add_scalar(f'{phase}/epoch_F1', epoch_f1, epoch)
            # epoch_iou = metrics['iou'] / epoch_samples
            # writer.add_scalar(f'{phase}/epoch_IOU', epoch_iou, epoch)

            # # deep copy the model
            # if phase == 'val' and epoch_loss < best_loss:
            #     print("saving best model")
            #     best_loss = epoch_loss
            #     best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    # model.load_state_dict(best_model_wts)
    return model


def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--policy', type=str, help='policy name')
    parser.add_argument('--run-folder', type=str, default='', help='folder to save in the tensorboard')
    parser.add_argument('--seed', type=int, help='random seed for dataloader')
    parser.add_argument('--img-dir', type=str, help='ct image dir')
    parser.add_argument('--mask-set-dir', type=str, help='masks set')
    parser.add_argument('--config', type=str, help='policy name')
    args = parser.parse_args()
    return args


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    args = parse()
    policy_mapper = {'random': RandomLearner,
                     'uncertain_entropy': UncertainLearnerEntropy,
                     'variance': VarianceLearnerDropout,
                     }
    with open(args.config, 'r') as f:
        config_json = json.load(f)
    run_folder = args.run_folder
    seed = args.seed
    batch_size = config_json['train']['batch_size']
    n_epochs = config_json['train']['n_epochs']
    lr = config_json['train']['lr']
    warmup_n_epochs = config_json['policy']['warmup']['n_epochs']
    frequency = config_json['policy']['update_frequency']
    n_batches_per_step = config_json['policy']['update_size']

    policy_class = policy_mapper.get(args.policy, None)
    assert policy_class is not None, 'bad policy name'

    img_dir = args.img_dir
    set_dir = args.mask_set_dir
    mask_dir_warmup = os.path.join(set_dir, 'warmup')
    mask_dir_pool = os.path.join(set_dir, 'pool')
    mask_dir_val = os.path.join(set_dir, 'test')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    trans = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.type(torch.FloatTensor),
    ])

    warmup_set = LungsDataset(img_dir, mask_dir_warmup, transform=trans, seed=seed)
    pool_set = LungsDataset(img_dir, mask_dir_pool, transform=trans, seed=seed)
    val_set = LungsDataset(img_dir, mask_dir_val, transform=trans, seed=seed)

    # train_sampler = ResumableRandomSampler(pool_set)
    dataloaders = {
        'train': {
            'warmup': DataLoader(warmup_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
            'pool': DataLoader(pool_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            # 'pool': DataLoader(pool_set, batch_size=batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)
        },
        'val': DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }

    in_channels = 1
    model = hub.load('mateuszbuda/brain-segmentation-pytorch',
                     'unet',
                     in_channels=in_channels,
                     out_channels=1,
                     init_features=32,
                     pretrained=False)
    model.load_state_dict(torch.load(f'./tmp/unet_{in_channels}.pth'))
    # torch.save(model.state_dict(), f'./tmp/unet_{in_channels}.pth')
    model = model.to(device)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    optimizer.load_state_dict(torch.load(f'./tmp/optimizer_{in_channels}.pth'))
    # torch.save(optimizer.state_dict(), f'./tmp/optimizer.pth')

    policy_learner = policy_class(dataloaders['train'], warmup_n_epochs, n_batches_per_step,
                                  model, update_freq=frequency,
                                  seed=seed, device=device)

    # writer
    cur_date = datetime.now().strftime('%Y_%m_%d_%H_%M')
    class_name = policy_learner.__class__.__name__.lower()
    set_name = basename(set_dir)
    os.makedirs(run_folder, exist_ok=True)
    writer_name = f'{run_folder}/{set_name}_{class_name}_{seed}_{cur_date}'

    model = train_model(model, dataloaders, policy_learner,
                        optimizer, None, num_epochs=n_epochs,
                        device=device, writer_name=writer_name)


if __name__ == '__main__':
    main()
