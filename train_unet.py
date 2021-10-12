""" # noqa
   ___           __________________  ___________
  / _/__  ____  / __/ ___/  _/ __/ |/ / ___/ __/
 / _/ _ \/ __/ _\ \/ /___/ // _//    / /__/ _/      # noqa
/_/ \___/_/   /___/\___/___/___/_/|_/\___/___/      # noqa
Author : Benjamin Blundell - k1803390@kcl.ac.uk

train_unet.py - train our u-net model - our main entry point.

"""
from comet_ml import Experiment
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
from util.image import reduce_source, reduce_mask, reduce_result
from codecarbon import EmissionsTracker
import wandb


def loss_func(result, target) -> torch.Tensor:
    # We weight the loss as most of the image is 0, or background.
    # TODO also, float32 isn't working well with this loss, unless I ignore gradients on the 0 class
    class_weights = torch.tensor(
        [0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=result.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # TODO - adjusted the permute here. Not sure if that's going to be correct.
    loss = criterion(result, target) + dice_loss(F.softmax(result, dim=1).float(),
                                                F.one_hot(target, model.n_classes).permute(
       0, 4, 1, 2, 3).float(), multiclass=True)

    #loss = criterion(result, target)
    return loss


def test(args, model, test_data: DataLoader, step: int, writer: SummaryWriter):
    model.eval()

    with torch.no_grad():
        source, target_mask = next(iter(test_data))
        result = model.forward(source)
        target_mask = target_mask.to(device=result.device, dtype=torch.long)
        loss = loss_func(result, target_mask)
        print('Test Step: {}.\tLoss: {:.6f}'.format(step, loss))

        # write to tensorboard
        writer.add_image('test_source_image', reduce_source(source), step)
        writer.add_image('test_source_image_side', reduce_source(source, 2), step)
        writer.add_image('test_target_image', reduce_mask(target_mask), step)
        writer.add_image('test_target_image_side', reduce_mask(target_mask, 1), step)
        writer.add_image('test_predict_image', np.expand_dims(reduce_result(result), axis=0), step)
        writer.add_image('test_predict_image_side', np.expand_dims(reduce_result(result, 1), axis=0), step)
        writer.add_scalar('test loss', loss, step)

        # WandB write
        wandb.log({'test_loss': loss})

        # Matches these of the input data
        class_labels = {
            0: "background",
            1: "ASI 1",
            2: "ASI 2",
            3: "ASJ 1",
            4: "ASJ 2"
        }

        classes = result[0].max(dim=0)[0].cpu()
        part_reduced = classes.amax(axis=0).unsqueeze(dim=0).squeeze().numpy()
        target_reduced = target_mask[0].amax(axis=0).cpu().unsqueeze(dim=0).squeeze().numpy()
   
        masked_image = wandb.Image(reduce_source(source).squeeze(), masks={
            "predictions": {
                "mask_data": part_reduced,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": target_reduced,
                "class_labels": class_labels
            }
        })

        wandb.log({"image_with_masks": masked_image})

        classes = result[0].max(dim=1)[0].cpu()
        part_reduced = classes.amax(axis=1).unsqueeze(dim=0).squeeze().numpy()
        target_reduced = target_mask[0].amax(axis=1).cpu().unsqueeze(dim=0).squeeze().numpy()
   
        masked_image_side = wandb.Image(reduce_source(source, 2).squeeze(), masks={
            "predictions": {
                "mask_data": part_reduced,
                "class_labels": class_labels
            },
            "ground_truth": {
                "mask_data": target_reduced,
                "class_labels": class_labels
            }
        })

        wandb.log({"image_with_masks_side": masked_image_side})

    model.train()


def evaluate(args, model, data: DataLoader):
    model.eval()
    num_batches = len(data)
    loss_total = 0

    # class_weights = torch.tensor(
    #    [0.1, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32, device=model.device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    with torch.no_grad():
        for batch in tqdm(data, total=num_batches, desc='Evaluation round', unit='batch', leave=False):
            source, target_mask = next(iter(data))
            result = model.forward(source)
            target_mask = target_mask.to(device=result.device, dtype=torch.long)
            mask_true = F.one_hot(target_mask, model.n_classes).permute(0, 4, 1, 2, 3).float()
            mask_pred = F.one_hot(result.argmax(dim=1), model.n_classes).permute(0, 4, 1, 2, 3).float()
            # compute the Dice score, ignoring background
            loss_total += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
            # loss_total += criterion(result, target_mask)

    model.train()
    return loss_total / num_batches


def train(args, model, train_data: DataLoader, test_data: DataLoader,  valid_data: DataLoader,
          optimiser, scheduler, writer: SummaryWriter):
    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Total trainable params:", pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Total params:", pytorch_total_params)
    model.train()

    # CO2 tracker
    tracker = EmissionsTracker(project_name="sea_elegance", output_dir=args.savedir, save_to_file=True)

    # Weights and Biases start
    experiment = wandb.init(project='sea_elegance',
                            resume='allow', entity='oni')
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr))

    wandb.watch(model)
    tracker.start()

    # Now start the training proper
    for epoch in range(args.epochs):
        for batch_idx, (source, target_mask) in enumerate(train_data):
            optimiser.zero_grad()
            result = model(source)
            target_mask = target_mask.to(device=result.device, dtype=torch.long)
            # TODO not sure the permute is right here?
            #loss = loss_func(result, target_mask)
            loss = loss_func(result, target_mask) + dice_loss(F.softmax(result, dim=1).float(),
                                                                F.one_hot(target_mask, model.n_classes).permute(
                                                                    0, 4, 1, 2, 3).float(),
                                                                multiclass=True)
            loss.backward()
            optimiser.step()
            step = epoch * len(train_data) + (batch_idx * args.batch_size)
            writer.add_scalar('training loss', float(loss), step)
            print('Train Epoch / Step: {} {}.\tLoss: {:.6f}'.format(epoch,
                                                                    batch_idx, float(loss)))
            wandb.log({'training_loss': loss})
          
            if batch_idx % args.log_interval == 0 and batch_idx != 0:
                save_checkpoint(model, optimiser, epoch, batch_idx,
                                loss, args, args.savedir, args.savename)
                test(args, model, test_data, step, writer)
                # Run a validation pass, with the scheduler
                wandb.log({'learning_rate': optimiser.param_groups[0]['lr']})

            del loss

            if batch_idx % args.save_interval == 0 and batch_idx != 0:
                save_model(model, args.savedir + "/model.tar")

            del source, target_mask
            torch.cuda.empty_cache()
        scheduler.step(evaluate(args, model, valid_data))

    tracker.stop()


def dataset_to_disk(args, dataset, filename="dataset.csv"):
    with open(args.savedir + "/" + filename, "w") as f:
        for (source, target) in dataset.img_targets.iloc:
            f.write(source + ", " + target + "\n")


def load_data(args, device) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Do we need device? Moving the data onto the device for speed?
    worm_data = WormDataset(annotations_file=args.image_path + "/dataset.csv",
                            img_dir=args.image_path,
                            device=device)
    dsize = args.train_size + args.test_size + args.valid_size
    assert(dsize <= len(worm_data))
    train_dataset, test_dataset, valid_dataset = torch.utils.data.random_split(
        worm_data, [args.train_size, args.test_size, args.valid_size])
    # Write out the datasets to files so we know which data were used in which
    # sets for later analysis
    dataset_to_disk(args, train_dataset, "dataset_train.csv")
    dataset_to_disk(args, test_dataset, "dataset_test.csv")
    dataset_to_disk(args, valid_dataset, "dataset_valid.csv")

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    return (train_dataloader, test_dataloader, valid_dataloader)


def create_model(args, device) -> NetU:
    model = NetU()
    model.device = device
    model.to(device)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Shaper Train')
    parser.add_argument('--batch-size', type=int, default=2,
                        metavar='N', help='input batch size for training (default: 2)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    optimiser = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, 'max', patience=2)  # goal: maximize Dice score

    # Start Tensorboard
    writer = SummaryWriter(args.savedir + '/experiment_tensorboard')

    # Start Comet for the co2 tracking
    # https://www.comet.ml
    # Create an experiment with your api key
    experiment = Experiment(
        api_key="qaUluNTHjUMaGs6gOkbo4VotI",
        project_name="sea_elegance",
        workspace="onidaito",
    )

    # Start training
    train(args, model, train_data, test_data,
          valid_data, optimiser, scheduler, writer)

    # Final things to write to tensorboard
    #images, _, _ = next(iter(train_data))
    #writer.add_graph(model, images)
    writer.close()
