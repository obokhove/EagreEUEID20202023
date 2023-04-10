## Instruction to install Firedrake in a Singularity docker on ARC4.

[OB et al. To do: Describe singularity. Virtual environment. HPC provides module singularity.
Onno 09-04-2023: I can virtually not follow a word: from which directory & machine are these commands given?
Poor editting as well, as usual, which I started to fix -my, my.]


Junho's instructions to date 09-04-2023:

1.	Make directories to save a Singularity docker.
  ```Python
  $cd
  $mkdir  /nobackup/cscmaw
  $mkdir  /nobackup/cscmaw/$USER
  $touch  .singularity
  $ln -s /nobackup/cscmaw/$USER/  .singularity
  $ mkdir /nobackup/cscmaw/firedrake
  
```
2.	Load modules required for installation.
```Python
 $ module swap openmpi mvapich
 $ module add apptainer
 $ module add anaconda
```
       
3.	Install Firedrake using Singularity
```Python
  $ cd /nobackup/cscmaw/firedrake
  $  apptainer pull docker://firedrakeproject/firedrake
  ```
4.	When finishing the installation, a Firedrake image named firedrake_latest.sif is made.
After that, move the Firedrake image into a directory where you want to compute.
```Python
   $mv firedrake_latest.sif /home/home02/$USER
```

5.	Now, the installation ends. To run a Firedrake code, there are two ways.

(a)	Command directly at a linux window. First, load Singularity as 
```Python
        $ module add singularity/3.6.4
```
  
  And then, use the following command:
  
```Python
$singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache firedrake_latest.sif python BL_test.py
```

(b)	Submit a job script to job scheduler of ARC4. An example of a job script is attached.

