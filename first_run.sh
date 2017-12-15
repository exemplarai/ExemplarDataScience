#!/bin/bash
: '
Recommend to install:
sudo apt-get install python3 python3-pip python3-dev virtualenvwrapper

'

virtualenv -p python3.6 exemplar
source exemplar/bin/activate
pip install -r requirements.txt


