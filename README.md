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

    python run.py --load ../runs/wormz_2021_10_04  --image /phd/wormz/queelim/dataset_21_10_2021/00618_layered.fits

## Evaluation

    python eval.py --load ../runs/wormz_2021_10_19 --data /media/proto_backup/wormz/queelim/dataset_18_10_2021

## Energy Usage
[Code Carbon](https://github.com/mlco2/codecarbon) is also used as part of our reporting on carbon emissions and energy usage.

## Visualisation

Using [Vedo](https://github.com/marcomusy/vedo) ([docs](https://vedo.embl.es/autodocs/content/vedo/index.html)) Once our viz.py is run.

    python viz/viz.py --image /media/proto_backup/wormz/queelim/dataset_21_10_2021/02374_layered.fits --savepath base.ply
    python viz/viz.py --image eval_0.fits --savepath predict.ply
    vedo base.ply predict.ply --alpha 0.5