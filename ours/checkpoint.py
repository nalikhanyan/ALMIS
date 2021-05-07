import os

import dill
import torch


def checkpoint_save(model, optimizer, learner, epoch, writer_name, n_images, filename='checkpoint.pth.tar'):
    used_ids = learner.used_ids if learner else set()
    print('---------------------', len(used_ids))
    state = {'epoch': epoch + 1, 'model': model.state_dict(), 'learner_ids': used_ids,
             'optimizer': optimizer.state_dict(), 'writer_name': writer_name, 'n_images': n_images}
    torch.save(state, filename)


def checkpoint_load(model, optimizer, learner, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        learner.update_used_points(checkpoint['learner_ids'])
        writer_name = checkpoint['writer_name']
        n_images = checkpoint['n_images']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, learner, start_epoch, writer_name, n_images


def scikit_model_save(model, file_name):
    with open(file_name, 'wb') as file_:
        dill.dump(model, file_)

def scikit_model_load(file_name):
    with open(file_name, 'rb') as file_:
        model = dill.load(file_)
    return model
