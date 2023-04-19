#!/bin/bash
# job standard output will go to the file slurm-%j.out (where %j is the job ID)

#SBATCH --time=23:15:00   # limit (HH:MM:SS)
#SBATCH --nodes=1  # number of nodes
#SBATCH --ntasks-per-node=40
#SBATCH --partition=compute
#SBATCH --mail-user=james.saunders@mail.utoronto.ca   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL

# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module load NiaEnv/2019b intelpython3
module load intel/2019u4
source activate pyMBGAN

/home/o/oespinga/jhsaunde/.conda/envs/pyMBGAN/bin/python gridsearch_ae.py