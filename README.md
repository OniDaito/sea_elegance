# Sea Elegance

## Training the network
Running the training:

    python train_unet.py --image-path ../wormz/queelim/dataset --train-size 41 --test-size 10

## Tests
Testing example: 
    python -m unittest test.data.Data

## Conda environment
Create a seaelegance conda setup:
    conda create -n seaelegance python

## Tensorboard
Tensorboard?
    tensorboard --logdir=save --bind_all

Navigate to:
    http://proto.local:6006/


## Running a prediction
Running a model with an image to make a prediction

    python run.py --load ../runs/wormz_2021_10_04  --image /phd/wormz/queelim/dataset_24_09_2021/00618_layered.fits

## External stats
This experiment links up with [https://www.comet.ml](Comet) and [Weights and Biases](wandb.ai).

## Energy Usage
[Code Carbon](https://github.com/mlco2/codecarbon) is also used as part of our reporting on carbon emissions and energy usage.