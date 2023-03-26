Hallo.

Describe singularity. Virtual environment. HPC provide module singularity. Where does on get this.


Instruction to install Firedrake in a Singularity docker on ARC4.

1.	Make directories to save a Singularity docker.
  $cd
  $mkdir  /nobackup/cscmaw
  $mkdir  /nobackup/cscmaw/$USER
  $touch  .singularity
  $ln -s /nobackup/cscmaw/$USER/  .singularity
  $ mkdir /nobackup/cscmaw/firedrake

2.	Load modules required for installation.
 $ module swap openmpi mvapich
 $ module add apptainer
 $ module add anaconda
       
3.	Install Firedrake using Singularity
$ cd /nobackup/cscmaw/firedrake
$  apptainer pull docker://firedrakeproject/firedrake
4.	When finishing the installation, a Firedrake image named firedrake_latest.sif is made.
After that, move the Firedrake image into a directory where you want to compute.
$mv firedrake_latest.sif /home/home02/$USER
[example.pdf](https://github.com/obokhove/EagreEUEID20202023/files/11071985/example.pdf)

5.	Now, the installation ends. To run a Firedrake code, there are two ways.
(a)	Command directly at a linux window. First, load Singularity as 
                     $ module add singularity/3.6.4 
                 And then, command the follows 
                  $singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache firedrake_latest.sif python BL_test.py
(b)	Submit a job script to job scheduler of ARC4. An example of a job script is attached.
