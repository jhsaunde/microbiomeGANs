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
module load gcc/8.3.0
module load r/4.1.2
module load cmake/3.21.4
module load mpfr/4.0.2
module load gmp/6.1.2

Rscript dataset/simulated_data/simulate_CVSparseDOSSA2.R