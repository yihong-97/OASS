#!/bin/bash

cd /path/to/the/unmaskformer
source ~/venv/unmaskformer/bin/activate
PYTHONPATH="</path/to/the/unmaskformer>:$PYTHONPATH" && export PYTHONPATH
python cityscapesscripts/preparation/createPanopticImgs.py