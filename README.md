# Compressing face images with autoencoders

The aim of this project is to experiment with deep learning autoencoders for coding images where there is typically a person in the foreground over a mostly static background. Our aim is to explore deep neural network architectures suitable for efficient communication on video conferencing use cases.

Our current approach is to simplify the video coding task by treating a video as a set of individual frames, rather than as a sequence. Accordingly, our dataset is a subset of individual frames collected from the VoxCeleb2 video dataset.

As of today, we provide a working PyTorch implementation of the deep learning architecture from the paper "Lossy image compression with compression autoencoders", by Lucas Theis, Wenzhe Shi, Andrew Cunningham & Ferenc Husz, published in 2017 (see the [original paper](https://arxiv.org/pdf/1703.00395v1.pdf) for details). While the original paper described a Theano implementation we have implemented and trained our own models in PyTorch, being able to reproduce the results from the paper.

This repo contains the PyTorch implementation and a training Jupyter notebook that downloads our dataset and pre-trained models, hosted in DropBox. The code was made open source in the hope of encouraging machine learning practitioners to use it as a baseline in their own research.

# Google Colab setup

The easiest way to experiment with the provided models is possibly to load compressor_train.ipynb in Google Colab. This workflow is especially useful if you only intend to browse the TensorBoards for the different models.

Open the file and run all the cells in order to download the dataset, trained models and TensorBoard logs.

# Workstation setup

If you intend to do more serious work you may want to setup your own development machine following the instructions from this section.

- Install opencv and jupyter packages on Ubuntu (or the equivalent for your preferred OS):

`sudo apt-get install python-opencv runipy`

- Install the needed python packages:

`pip3 install psutil scikit-image opencv-python pytest torchvision pandas tqdm torch`

- Double check you have CUDA support by checking the following command line prints "True" rather than "False":

`(echo "import torch"; echo "torch.cuda.is_available()") | python3 -i`

- Double check you have pytorch 1.5 installed in your system:

`(echo "import torch"; echo "torch.__version__") | python3 -i`

- Clone the repo:

`git clone git@github.com:abel-bernabeu/autoencoder.git`

- Change directory to "autoencoder":

`cd autoencoder`

- Run the training notebook to get the latest version of the dataset and trained models:

`runipy compressor_train.ipynb`

# Test suite

All the unit and integration tests are discovered with "pytest" introspection, so you just need to type one command for executing them all:

`pytest`
