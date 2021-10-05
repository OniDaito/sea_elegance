""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train_unet.py - train our u-net model - our main entry point.

"""
from torch.serialization import save
from torch.utils.tensorboard.summary import image_boxes
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
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch import autograd
from net.dice_score import dice_loss, multiclass_dice_coeff
from util.loadsave import save_checkpoint, save_model
from codecarbon import track_emissions
import wandb


def loss_func(result, target) -> torch.Tensor:
    # We weight the loss as most of the image is 0, or background.
    # TODO also, float16 isn't working well with this loss, unless I ignore gradients on the 0 class
    class_weights = torch.tensor(
        [0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float16, device=result.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # TODO - adjusted the permute here. Not sure if that's going to be correct.
    #loss = criterion(result, target)  + dice_loss(F.softmax(result, dim=1).float(),
    #                                            F.one_hot(target, model.n_classes).permute(
    #   0, 4, 1, 2, 3).float(), multiclass=True)

    loss = criterion(result, target)
    return loss


def reduce_source(image, axis=1) -> np.ndarray:
    m = torch.max(image).item()
    final = image.amax(axis=axis).cpu().numpy()
    return np.array(final / m * 255).astype(np.uint8)


def reduce_mask(image, axis=0) -> np.ndarray:
    final = image.amax(axis=axis).cpu().unsqueeze(dim=0).numpy()
    return np.array(final / 4 * 255).astype(np.uint8)


def convert_result(image, axis=0) -> np.ndarray:
    classes = image.max(dim=0)[0].cpu()
    final = classes.amax(axis=axis).unsqueeze(dim=0).numpy()
    return np.array(final / 4 * 255).astype(np.uint8)


def test(args, model, test_data: DataLoader, step: int, writer: SummaryWriter):
    model.eval()

    with torch.no_grad():
        source, target_mask = next(iter(test_data))
        result = model.forward(source)
        target_mask = target_mask.to(device=result.device, dtype=torch.long).to_dense()
        loss = loss_func(result, target_mask)
        print('Test Step: {}.\tLoss: {:.6f}'.format(step, loss))

        # write to tensorboard
        writer.add_image('test_source_image', reduce_source(source[0]), step)
        writer.add_image('test_source_image_side',
                         reduce_source(source[0], 2), step)
        writer.add_image('test_target_image', reduce_mask(
            target_mask[0]), step)
        writer.add_image('test_target_image_side', reduce_mask(
            target_mask[0], 1), step)
        writer.add_image('test_predict_image', convert_result(result[0]), step)
        writer.add_image('test_predict_image_side',
                         convert_result(result[0], 1), step)
        writer.add_scalar('test loss', loss, step)

    model.train()


def evaluate(args, model, data: DataLoader):
    model.eval()
    num_batches = len(data)
    #dice_score = 0
    loss_total = 0

    class_weights = torch.tensor(
        [0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float16, device=model.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for batch in tqdm(data, total=num_batches, desc='Evaluation round', unit='batch', leave=False):
            source, target_mask = next(iter(test_data))
            result = model.forward(source).to(dtype=torch.float32)
            target_mask = target_mask.to(device=result.device, dtype=torch.long).to_dense()
            #mask_true = F.one_hot(target_mask, model.n_classes).permute(0, 4, 1, 2, 3).float()
            #mask_pred = F.one_hot(result.argmax(dim=1), model.n_classes).permute(0, 4, 1, 2, 3).float()
            # compute the Dice score, ignoring background
            #dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            loss_total += criterion(result, target_mask)

    model.train()
    #return dice_score / num_batches
    return loss_total / num_batches


@track_emissions(project_name="sea_elegance")
def train(args, model, train_data: DataLoader, test_data: DataLoader,  valid_data: DataLoader,
          optimiser, scheduler, writer: SummaryWriter):
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Total trainable params:", pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", pytorch_total_params)
    model.train()

    # Weights and Biases start
    experiment = wandb.init(project='sea_elegance',
                            resume='allow', entity='oni')
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr))

    # Now start the training proper
    for epoch in range(args.epochs):

        for batch_idx, (source, target_mask) in enumerate(train_data):
            optimiser.zero_grad()
            result = model(source).to(dtype=torch.float32)
            target_mask = target_mask.to(device=result.device, dtype=torch.long).to_dense()
            # TODO not sure the permute is right here?
            loss = loss_func(result, target_mask)
            #loss = loss_func(result, target_mask) + dice_loss(F.softmax(result, dim=1).float(),
            #                                                  F.one_hot(target_mask, model.n_classes).permute(
            #                                                      0, 4, 1, 2, 3).float(),
            #                                                  multiclass=True)
            loss.backward()
            optimiser.step()
            step = epoch * len(train_data) + (batch_idx * args.batch_size)
            writer.add_scalar('training loss', float(loss), step)
            print('Train Epoch / Step: {} {}.\tLoss: {:.6f}'.format(epoch,
                                                                    batch_idx, float(loss)))

            # Run a validation pass, with the scheduler
            scheduler.step(evaluate(args, model, valid_data))

            if batch_idx % args.log_interval == 0 and batch_idx != 0:
                save_checkpoint(model, optimiser, epoch, batch_idx,
                                loss, args, args.savedir, args.savename)
                test(args, model, test_data, step, writer)

                # Weights and biases log
                histograms = {}
                for tag, value in model.named_parameters():
                    tag = tag.replace('/', '.')
                    histograms['Weights/' +
                               tag] = wandb.Histogram(value.data.cpu())
                    histograms['Gradients/' +
                               tag] = wandb.Histogram(value.grad.data.cpu())

                experiment.log({
                    'learning rate': optimiser.param_groups[0]['lr'],
                    # 'validation Dice': val_score,
                    'images': wandb.Image(source[0].cpu()),
                    'masks': {
                        'true': wandb.Image(target_mask[0].float().cpu()),
                        'pred': wandb.Image(torch.softmax(result, dim=1)[0].float().cpu()),
                    },
                    'step': batch_idx,
                    'epoch': epoch,
                    **histograms
                })

            del loss

            if batch_idx % args.save_interval == 0 and batch_idx != 0:
                save_model(model, args.savedir + "/model.tar")

            del source, target_mask
            torch.cuda.empty_cache()


def load_data(args, device) -> Tuple[DataLoader]:
    # Do we need device? Moving the data onto the device for speed?
    worm_data = WormDataset(annotations_file=args.image_path + "/dataset.csv",
                            img_dir=args.image_path,
                            device=device)
    #print("Length of Worm Dataset", len(worm_data), "on device", device)

    dsize = args.train_size + args.test_size + args.valid_size
    assert(dsize <= len(worm_data))
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(
        worm_data, [args.train_size, args.test_size, args.valid_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True)
    return (train_dataloader, test_dataloader, valid_dataloader)


def create_model(args, device) -> NetU:
    model = NetU()
    model.device = device
    model.to(device)
    return model


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
    parser.add_argument('--log-interval', type=int, default=10,
                        metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--savedir', default="./save",
                        help='The name for checkpoint save directory.')
    parser.add_argument("--savename", default="checkpoint.pth.tar",
                        help="The name for checkpoint save file.",
                        )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="how many batches to wait before saving.",
    )
    parser.add_argument('--image-path', default="",
                        help='Directory of images for training.')
    parser.add_argument('--train-size', type=int, default=22,
                        help='The size of the training set (default: 100)')
    parser.add_argument('--test-size', type=int, default=5,
                        help='The size of the training set (default: 20)')
    parser.add_argument('--valid-size', type=int, default=0,
                        help='The size of the training set (default: 10)')

    args = parser.parse_args()
    assert(args.image_path != "")

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Using device", device)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.cufft_plan_cache.max_size = 1024

    # Create all the things we need
    train_data, test_data, valid_data = load_data(args, device)
    model = create_model(args, device)
    # optimiser = optim.Adam(model.parameters(), lr=args.lr,
    #                       weight_decay=1e-5, eps=1e-3)
    # TODO - using float16 so maybe 1e-8 wont work?
    optimiser = optim.RMSprop(
        model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 'max', patience=2)  # goal: maximize Dice score

    # Start Tensorboard
    writer = SummaryWriter(args.savedir + '/experiment_tensorboard')

    # Start training
    train(args, model, train_data, test_data,
          valid_data, optimiser, scheduler, writer)

    # Final things to write to tensorboard
    #images, _, _ = next(iter(train_data))
    #writer.add_graph(model, images)
    writer.close()
