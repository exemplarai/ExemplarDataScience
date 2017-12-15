#!/bin/bash

[ ! -d "input" ] && mkdir input



source exemplar/bin/activate
export AWS_ACCESS_KEY_ID="key"
export AWS_SECRET_ACCESS_KEY="key"
export AWS_DEFAULT_REGION="us-west-2"


cd input
aws s3 sync s3://data-export.exemplar.ai .
cd ..


python main.py
python calculating.py
