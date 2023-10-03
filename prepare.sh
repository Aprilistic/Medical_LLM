#!/bin/sh

# Prepare the environment for the build
pip install -r requirements.txt

source .env

export WANDB_API_KEY=$WANDB_API_KEY
