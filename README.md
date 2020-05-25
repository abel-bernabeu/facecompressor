# autoencoder

This project is an experiment on using deep learning autoencoders for coding videos where there is typically a person in the foreground over a mostly static background. Our aim is to explore deep neural network architectures suitable for efficient video coding on video conferencing use cases.

# Environment setup

- Install the needed python packages

`pip3 install psutil`

- Double check you have CUDA support by checking the following command line prints "True" rather than "False"

`(echo "import torch"; echo "torch.cuda.is_available()") | python3 -i`

- Double check you have pytorch 1.5 is available in your system.

`(echo "import torch"; echo "torch.__version__") | python3 -i`

- Clone the repo:

`git clone git@github.com:abel-bernabeu/autoencoder.git`

- Change directory to "autoencoder":

`cd autoencoder`

- Get the YouTubers dataset`

`python3 download.py`

