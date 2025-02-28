#!/bin/bash

#SBATCH -o /PATH/TO/USER/DIRECTORY/scgpt/runs/%j/scgpt.log
#SBATCH -e /PATH/TO/USER/DIRECTORY/scgpt/runs/%j/scgpt.err
#SBATCH -p med-gpu
#SBATCH -N 1

# Replace with user's scratch directory
#SBATCH -D /scratch/ligrp/USER_NAME/scgpt
#SBATCH -J scGPT_Test
#SBATCH --export=None
#SBATCH -t 4:00:00
#SBATCH --mem=30G

# Replace with user's home directory 
HOME_DIR=/CHILDRENS/home/SYSTEM_OR_GW_ID/scgpt
RUN_DIR=$HOME_DIR/runs/${SLURM_JOB_ID}/
mkdir -p $RUN_DIR

# Set working directory 
WD=/scratch/ligrp/USER_NAME/scgpt
DATA_DIR=$WD/data

# Check if working directory exists
if [ -d "$WD" ]; then
    echo "Directory $WD exists. Aborting setup."
else
    # Set up working directory
    echo "Directory $WD does not exist. Initiating setup."
    mkdir -p $WD

    echo "created WD: $WD"
    cd $WD

    # Download scGPT via uv pip
    git clone https://github.com/bowang-lab/scGPT.git $WD/scGPT
    # git clone https://github.com/zacheliason/scGPT.git 
    mv $WD/scGPT/* $WD/
    rm -rf $WD/scGPT

    # Make sure a copy of pyproject.toml exists in the working directory
    cp $HOME_DIR/pyproject.toml $WD/pyproject.toml

    # Download and set up UV (UV is a really fast Python package and project manager)
    export UV_ROOT=$WD/.uv
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH=$UV_ROOT/bin:$PATH

    # Create virtual environment
    uv python install 3.10
    uv venv $WD/.venv --python 3.10
    uv pip install -r pyproject.toml
    source $WD/.venv/bin/activate

    # Download data
    mkdir -p $DATA_DIR
    gdown --folder https://drive.google.com/drive/folders/1oWh_-ZRdhtoGQ2Fw24HP41FgLoomVo-y -O $DATA_DIR/scGPT_human

    echo "Download complete. Files saved to: $DATA_DIR"
fi

echo "Running scGPT on data in $DATA_DIR and saving results to $RUN_DIR"
source $WD/.venv/bin/activate && python3 scgpt_perturb.py -d $DATA_DIR -o $RUN_DIR
