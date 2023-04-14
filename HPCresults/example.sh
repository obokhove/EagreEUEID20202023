#!/bin/bash

#-P feps-cpu
#$ -l nodes=1
#$ -cwd -V
#$ -l h_rt=0:10:0
#$ -l node_type=40core-192G
#$ -pe smp 1

module swap openmpi mvapich2
module add apptainer

singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache firedrake_latest.sif mpiexec -n 40 python pot.py
