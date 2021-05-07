import argparse
import json
import os
import random
import time
from collections import defaultdict
from copy import deepcopy
from ntpath import basename

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from dataloaders.ct_lungs import LungsDataset, SimpleDataset
from metrics import calc_f1, dice_loss, print_metrics
from policy import RandomLearner
from resumable_sampler import ResumableRandomSampler
from torch import hub
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm
from unet_drp import get_drp_unet

from ours.checkpoint import checkpoint_load, checkpoint_save


def get_loss_score(model, loader, device):
    trainig = model.training
    model.eval()
    loss_sum = 0
    acc_sum = 0
    num_pts = 0
    for _, inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = dice_loss(outputs, labels)
            acc = calc_f1(outputs, labels)
        loss_sum += loss * inputs.size(0) 
        acc_sum += acc * inputs.size(0)
        num_pts += inputs.size(0)
    model.train() if model.training else model.eval()
    return loss_sum / num_pts, acc_sum / num_pts


def train_model(model, dataloaders, policy_learner, optimizer, scheduler, start_epoch, num_epochs, device, writer, n_images=None):
    loader = {'val': dataloaders['val']}

    # best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    if n_images is None:
        n_images = {'train': 0, 'val': 0}
    new_points = []

    for epoch in range(start_epoch, start_epoch+num_epochs):
        loader['train'] = policy_learner()
        new_points += sum([list(zip(*batch))
                           for batch in policy_learner.batches], [])
        print('Epoch {}/{}'.format(epoch, start_epoch+num_epochs - 1))
        print('-' * 10)

        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            # print('+++++++++ len loader', len(loader[phase]))
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
                    loss = dice_loss(outputs, labels)
                    acc_f1 = calc_f1(outputs, labels)
                    # acc_iou = calc_IOU(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)
                n_images[phase] += inputs.size(0)

                writer.add_scalar(f'{phase}/loss', loss.data.cpu().numpy(), n_images[phase])
                # writer.add_scalar(f'{phase}/bce', loss_bce, n_images[phase])
                # writer.add_scalar(f'{phase}/dice', loss_dice, n_images[phase])

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
    return model, n_images, new_points


def train_single(model, train_batch, loader_val, optimizer, device, out_dir):
    old_loss, old_f1 = get_loss_score(model, loader_val, device)
    _, inputs, labels = train_batch
    inputs = inputs.to(device)
    labels = labels.to(device)

    # zero the parameter gradients
    optimizer.zero_grad()

    with torch.set_grad_enabled(True):
        outputs = model(inputs)
        loss = dice_loss(outputs, labels)

        # backward + optimize only if in training phase
        loss.backward()
        optimizer.step()

    new_loss, new_f1 = get_loss_score(model, loader_val, device)
    pred_batch = model(inputs)
    torch.save(pred_batch, out_dir)
    return out_dir, (old_loss - new_loss).cpu().item(), (new_f1 - old_f1).cpu().item()


def gen_data(num, model, points, loader_val, optimizer, device, batch_size, out_dir=None):
    print('=====================')
    print('===== Generating ====', batch_size)
    gen_dataset = SimpleDataset(points)
    gen_loader = DataLoader(gen_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0, pin_memory=True)
    data = []
    for i in tqdm(range(num // len(gen_loader))):
        gen_loader = DataLoader(gen_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0, pin_memory=True)
        for j, batch in enumerate(gen_loader):
            model_t = deepcopy(model)
            optimizer_t = torch.optim.Adam(model_t.parameters())
            optimizer_t.load_state_dict(optimizer.state_dict())
            out_file = os.path.join(out_dir, f'preds_{i}_{j}')
            res = train_single(model_t, batch, loader_val, optimizer_t, device, out_file)
            data.append(res)
    return data


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, help='policy name')
    parser.add_argument('--seed', type=int, help='random seed for dataloader')
    parser.add_argument('--saved-model', type=str, help='saved model path')
    parser.add_argument('--gen-dir', type=str, help='generated features directory')
    parser.add_argument('--run-folder', type=str, help='folder for writer')
    parser.add_argument('--img-dir', type=str, help='ct image dir')
    parser.add_argument('--out-dir', type=str, help='saved model output directory')
    parser.add_argument('--mask-set-dir', type=str, help='masks set')
    args = parser.parse_args()
    return args


def main():
    args = parse()
    out_dir = args.out_dir
    run_folder = args.run_folder
    seed = args.seed
    with open(args.config, 'r') as f:
        config_json = json.load(f)
    gen_dir = args.gen_dir
    batch_size = config_json['train']['batch_size']
    n_epochs = config_json['policy']['gen']['n_epochs']
    lr = config_json['train']['lr']
    frequency = config_json['policy']['update_frequency']
    n_batches_per_step = config_json['policy']['update_size']
    gen_num = config_json['policy']['gen']['num']

    img_dir = args.img_dir
    set_dir = args.mask_set_dir
    mask_dir_warmup = os.path.join(set_dir, 'warmup')
    mask_dir_pool = os.path.join(set_dir, 'pool')
    mask_dir_val = os.path.join(set_dir, 'test')
    mask_dir_val_learn = os.path.join(set_dir, 'val_learn')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(seed)

    trans = transforms.Compose([
        transforms.ToTensor(),
        lambda x: x.type(torch.FloatTensor),
    ])

    warmup_set = LungsDataset(img_dir, mask_dir_warmup, transform=trans, seed=seed, is_rgb=True)
    pool_set = LungsDataset(img_dir, mask_dir_pool, transform=trans, seed=seed, is_rgb=True)
    val_learn_set = LungsDataset(img_dir, mask_dir_val_learn, transform=trans, seed=seed, is_rgb=True)
    test_set = LungsDataset(img_dir, mask_dir_val, transform=trans, seed=seed, is_rgb=True)
    # train_sampler = ResumableRandomSampler(pool_set)
    dataloaders = {
        'train': {
            'warmup': DataLoader(warmup_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True),
            'pool': DataLoader(pool_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
            # 'pool': DataLoader(pool_set, batch_size=batch_size, shuffle=False, num_workers=0, sampler=train_sampler, pin_memory=True)
        },
        'val': DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    }

    model = get_drp_unet(3, True)
    model.to(device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    policy_learner = RandomLearner(dataloaders['train'], 0, n_batches_per_step, model,
                                   update_freq=frequency, seed=seed, shuffle=True, device=device)

    model, optimizer, policy_learner, start_epoch, writer_name, n_images = checkpoint_load(
        model, optimizer, policy_learner, args.saved_model)
    model.to(device)
    model.train()
    model_warmuped = deepcopy(model)

    writer = SummaryWriter(writer_name, flush_secs=1)
    writer.close()

    model_trained, n_images, new_points = train_model(
        model, dataloaders, policy_learner, optimizer, None, start_epoch, n_epochs, device, writer, n_images)

    wrt_name = basename(writer_name)
    filename = os.path.join(out_dir, f'gen_{wrt_name}.pt')
    checkpoint_save(model, optimizer, policy_learner, start_epoch + n_epochs-1, writer_name, n_images, filename)

    val_learn_loader = DataLoader(val_learn_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    data = gen_data(gen_num, model_warmuped, new_points, val_learn_loader, optimizer,
                    device, batch_size, out_dir=gen_dir)
    pd.DataFrame(data, columns=['path', 'diff_loss', 'diff_f1']).to_csv(os.path.join(gen_dir, 'gt.csv'), index=False)


if __name__ == '__main__':
    main()
