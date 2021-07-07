""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train_unet.py - train our u-net model - our main entry point.

"""
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from data.loader import WormDataset
from typing import List, Tuple
from net.unet import NetU


def loss_func(result, target) -> torch.Tensor:
    # return F.l1_loss(result, target, reduction="sum")
    criterion = nn.BCEWithLogitsLoss()
    return criterion(result, target)


def test(args, model, test_data: DataLoader):
    model.eval()

    for idx, data in enumerate(test_data):
        (source, target_asi, _) = data
        result = model.forward(source)
        loss = loss_func(result, target_asi)
        print('Test Step: {}.\tLoss: {:.6f}'.format(idx, loss))

        '''
        for jdx in range(batch_in):
            save_image(batch_in[jdx][0],\
                "log/input_e" + str(epoch).zfill(3) +\
                "_i" + str(jdx).zfill(3) + ".png")
            save_image(batch_out[jdx][0].float(),\
                "log/target_e" + str(epoch).zfill(3) +\
                "_i" + str(jdx).zfill(3) + ".png")
            bin_img = torch.sigmoid(result[jdx][0]) > 0.5
            save_image(bin_img.float(), "log/output_e" +\
                str(epoch).zfill(3) +\
                "_i" + str(jdx).zfill(3) + ".png")
        '''


def train(args, model, train_data: DataLoader, test_data: DataLoader, optimiser):
    model.train()

    for epoch in range(args.epochs):
        for batch_idx, (source, target_asi, _) in enumerate(train_data):
            optimiser.zero_grad()
            result = model(source)
            loss = loss_func(result, target_asi)
            loss.backward()
            optimiser.step()

            print(
                'Train Epoch / Step: {} {}.\tLoss: {:.6f}'.format(epoch, batch_idx, loss))

            # We save here because we want our first step to be untrained
            # network
            if batch_idx % args.log_interval == 0:
                test(args, model, test_data)
                model.train()


def binaryise(input_tensor: torch.Tensor) -> torch.Tensor:
    ''' Convert the tensors so we don't have different numbers. If its
    not a zero, it's a 1.'''
    res = input_tensor.clone()
    res[input_tensor != 0] = 1
    return res


def load_data(args) -> Tuple[DataLoader]:
    # Do we need device? Moving the data onto the device for speed?
    worm_data = WormDataset(annotations_file=args.image_path + "/dataset.csv",
                            img_dir=args.image_path,
                            target_transform=binaryise)
    print("Length of Worm Dataset", len(worm_data))
    assert(args.train_size + args.test_size +
           args.valid_size <= len(worm_data))
    train_dataset, test_dataset = torch.utils.data.random_split(
        worm_data, [args.train_size, args.test_size])
    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=True)
    return (train_dataloader, test_dataloader)


def create_model(args, device) -> NetU:
    model = NetU()
    model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Shaper Train')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for training (default: 2)')
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
    train_data, test_data = load_data(args)
    model = create_model(args, device)
    variables = list(model.parameters())
    optimiser = optim.Adam(variables, lr=args.lr)

    # Start training
    train(args, model, train_data, test_data, optimiser)
