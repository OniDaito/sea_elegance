""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train_unet.py - train our u-net model - our main entry point.

"""
from torch.utils.tensorboard.summary import image_boxes
from net.guru import GuruMeditation
import torch
import argparse
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from data.loader import WormDataset
from typing import List, Tuple
from net.unet import NetU
import matplotlib.pyplot as plt
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
import pdb


def matplotlib_imshow(img):
    img = image_boxes.astype("int8")
    plt.imshow(img, cmap='jet')


def binaryise(input_tensor: torch.Tensor) -> torch.Tensor:
    ''' Convert the tensors so we don't have different numbers. If its
    not a zero, it's a 1.'''
    res = input_tensor.clone()
    res[input_tensor != 0] = 1
    return res


def loss_func(result, target) -> torch.Tensor:
    # return F.l1_loss(result, target, reduction="sum")
    criterion = nn.BCEWithLogitsLoss()
    dense = target.cpu().to(result.device)
    print(result.is_sparse, dense.is_sparse)
    return criterion(result, dense)


def reduce_image(image, axis=1) -> np.ndarray:
    return np.max(image.cpu().numpy().astype(float), axis=axis)


def test(args, model, test_data: DataLoader, step: int, writer: SummaryWriter):
    model.eval()
    source, target_asi, _ = next(iter(test_data))
    result = model.forward(source)
    loss = loss_func(result, target_asi)
    print('Test Step: {}.\tLoss: {:.6f}'.format(step, loss))

    # create grid of images for tensorboard
    # 16 bit int image maximum really so use that range
    # Only showing the first of the batch as we have 3D images, so we are going with 2D slices
    #source_grid = torchvision.utils.make_grid(reduce_image(source), normalize=True, value_range=(0, 4095))
    #source_grid_side = torchvision.utils.make_grid(reduce_image(source, 3), normalize=True, value_range=(0, 4095))
    # Pass output through a sigmnoid for single class prediction
    sigged = torch.sigmoid(result)
    gated = torch.gt(sigged, 0.5)
    final = gated.int()

    # write to tensorboard
    writer.add_image('test_source_image', reduce_image(source[0]), step)
    writer.add_image('test_source_image_side', reduce_image(source[0], 2), step)
    writer.add_image('test_target_image', reduce_image(target_asi[0]), step)
    writer.add_image('test_target_image_side', reduce_image(target_asi[0], 2), step)
    writer.add_image('test_predict_image', reduce_image(final[0]), step)
    writer.add_image('test_predict_image_side', reduce_image(final[0], 2), step)
    writer.add_scalar('test loss', loss, step)


def train(args, model, train_data: DataLoader, test_data: DataLoader, optimiser, writer: SummaryWriter):
    model.train()

    for epoch in range(args.epochs):
        for batch_idx, (source, target_asi, _) in enumerate(train_data):
            optimiser.zero_grad()
            result = model(source)
            loss = loss_func(result, target_asi)
            loss.backward()
            # Nicked from U-net example - not sure why
            #nn.utils.clip_grad_value_(model.parameters(), 0.1)
            optimiser.step()
            step = epoch * len(train_data) + (batch_idx * args.batch_size)
            writer.add_scalar('training loss', loss, step)
            print(
                'Train Epoch / Step: {} {}.\tLoss: {:.6f}'.format(epoch, batch_idx, loss))

            # We save here because we want our first step to be untrained
            # network
            if batch_idx % args.log_interval == 0:
                save(args, model)
                test(args, model, test_data, step, writer)
                model.train()


def load_data(args, device) -> Tuple[DataLoader]:
    # Do we need device? Moving the data onto the device for speed?
    worm_data = WormDataset(annotations_file=args.image_path + "/dataset.csv",
                            img_dir=args.image_path,
                            device=device)
    print("Length of Worm Dataset", len(worm_data))
   
    dsize = args.train_size + args.test_size + args.valid_size
    assert(dsize <= len(worm_data))
    train_dataset, test_dataset, _ = torch.utils.data.random_split(
        worm_data, [args.train_size, args.test_size, len(worm_data) - dsize])
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
    return (train_dataloader, test_dataloader)


def create_model(args, device) -> NetU:
    model = NetU()
    model.device = device
    model.to(device)
    return model


def save(args, model):
    torch.save(model, args.savedir + '/model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Shaper Train')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--image-size', type=int, default=512, metavar='S',
                        help='Assuming square imaages, what is the size? (default: 512)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100,
                        metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--savedir', default="./save",
                        help='The name for checkpoint save directory.')
    parser.add_argument('--image-path', default="",
                        help='Directory of images for training.')
    parser.add_argument('--train-size', type=int, default=22,
                        help='The size of the training set (default: 22)')
    parser.add_argument('--test-size', type=int, default=5,
                        help='The size of the training set (default: 5)')
    parser.add_argument('--valid-size', type=int, default=0,
                        help='The size of the training set (default: 0)')

    args = parser.parse_args()
    assert(args.image_path != "")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Create all the things we need
    train_data, test_data = load_data(args, device)
    model = create_model(args, device)
    # Adam optimiser results in NaNs which is a real shame
    optimiser = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5, eps=1e-3)
    #optimiser = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    #optimiser = optim.Rprop(model.parameters(), lr=args.lr)

    # Start Tensorboard
    writer = SummaryWriter(args.savedir + '/experiment_tensorboard')

    # Start training
    train(args, model, train_data, test_data, optimiser, writer)

    # Final things to write to tensorboard
    images, _, _ = next(iter(train_data))
    writer.add_graph(model, images)
    writer.close()
