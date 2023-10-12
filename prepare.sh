#!/bin/sh

DIR="$( cd "$( dirname "$0" )" && pwd )"

# Prepare the environment for the build
pip install --upgrade pip
pip install -r "$DIR/requirements.txt"
