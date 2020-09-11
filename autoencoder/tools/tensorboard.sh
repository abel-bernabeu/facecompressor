#!/bin/bash

tensorboard --logdir share/experiment1/runs --port 6100 &
tensorboard --logdir share/experiment2/runs --port 6200 &
tensorboard --logdir share/experiment3/runs --port 6300 &
tensorboard --logdir share/experiment4/runs --port 6400 &
tensorboard --logdir share/experiment5/runs --port 6500 &
