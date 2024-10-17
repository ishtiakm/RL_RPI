#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found, please install Anaconda or Miniconda first."
    exit
fi

# Create or update the conda environment
if [ -f environment.yml ]; then
    echo "Creating the conda environment from environment.yml..."
    conda env create -f environment.yml
else
    echo "No environment.yml found, exporting the current environment from history..."
    conda env export --from-history > environment.yml
    conda env create -f environment.yml
fi

# Activate the environment
echo "Activating the environment 'my_mdp'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate my_mdp

# Install pip packages (if requirements.txt is provided)
if [ -f requirements.txt ]; then
    echo "Installing additional pip packages from requirements.txt..."
    pip install -r requirements.txt
fi

# Run the Python script
echo "Running handwritten.py..."
python3 cliff_main.py

# Confirm execution completion
echo "Execution of cliff_main.py is complete."
