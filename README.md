# Face autoencoder

This project is an experiment on using deep learning autoencoders for coding videos where there is typically a person in the foreground over a mostly static background. Our aim is to explore deep neural network architectures suitable for efficient video coding on video conferencing use cases.

# Environment setup

- Install opencv and jupyter packages on Ubuntu (or the equivalent for your preferred OS):

`sudo apt-get install python-opencv jupyter`

- Install the needed python packages:

`pip3 install psutil scikit-image opencv-python pytest`

- Double check you have CUDA support by checking the following command line prints "True" rather than "False":

`(echo "import torch"; echo "torch.cuda.is_available()") | python3 -i`

- Double check you have pytorch 1.5 installed in your system:

`(echo "import torch"; echo "torch.__version__") | python3 -i`

- Clone the repo:

`git clone git@github.com:abel-bernabeu/autoencoder.git`

- Change directory to "autoencoder":

`cd autoencoder`

- Get the YouTubers dataset:

`python3 tools/downloader.py`

Optionally, if you are using PyCharm you may want to also follow the GitHub setup instructions from the JetBrains website:

https://www.jetbrains.com/help/pycharm/github.html


# Test suite

All the unit and integration tests are discovered with "pytest" instrospection, so you just need to type one command for executing them all:

`pytest`
