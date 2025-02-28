### scGPT Slurm Job Submission on HPC

This repository contains the necessary scripts to run scGPT on a High Performance Computing (HPC) cluster using Slurm. The scripts are designed to be used with the [scGPT](https://github.com/bowang-lab/scGPT) repository.

### Usage
You can use the following command to submit a job to the HPC cluster:
```bash
sbatch slurm_script.sh
```

The `slurm_script.sh` file will check if the required software is installed and will proceed to set up the working directory if it is not. This includes: 

- Cloning the scGPT repository
- Downloading the necessary data
- Setting up the Python environment (using UV)
    - (make sure `pyproject.toml` is in the same `$HOME_DIRECTORY` referenced in `slurm_script.sh`—this file is used to install the necessary Python packages with all the correct versions)

Then, the script will submit the job to the HPC cluster. 