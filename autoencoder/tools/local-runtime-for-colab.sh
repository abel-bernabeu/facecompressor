#!/bin/sh

# This script follows the instructions from:
#
# https://research.google.com/colaboratory/local-runtimes.html

pip3 install jupyter_http_over_ws
jupyter serverextension enable --py jupyter_http_over_ws
jupyter notebook \
  --NotebookApp.allow_origin='https://colab.research.google.com' \
  --port=8888 \
  --NotebookApp.port_retries=0
