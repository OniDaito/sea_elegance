# Sea Elegance

Running the training:

    python train_unet.py --image-path ../wormz/queelim/dataset --train-size 41 --test-size 10

Testing example: 
    python -m unittest test.data.Data

Create a seaelegance conda setup:
    conda create -n seaelegance python

Tensorboard?
    tensorboard --logdir=save --bind_all

Navigate to:
    http://proto.local:6006/


Running a model with an image to make a prediction

    python run.py --load ../runs/wormz_2021_10_04  --image /media/proto_backup/wormz/queelim/dataset_24_09_2021/00618_layered.fits