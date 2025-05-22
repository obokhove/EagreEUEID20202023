## Instruction to install Firedrake in a Singularity docker on HPC-ARC4 13-04-2023

By Junho, 감사합니다 (kamza hamnida), with OB

The former 
```Python
/nobackup/$USER/firedrake
```
is where Singularity docker for Firedrake is installed, the latter
```Python
/home/home02/$USER
```
is where you run Firedrake. So when you install Singularity, go to the former, type the commands at the former directory. To run Firedrake, go to the latter directory. 

Instruction steps:

0. Log onto the ARC4 HPC, see: https://arcdocs.leeds.ac.uk/getting_started/logon/logon-off-campus.html#connecting-from-linux-macos-systems

To run larger jobs use queu:
https://arcdocs.leeds.ac.uk/usage/batchjob.html![image](https://user-images.githubusercontent.com/16267535/230985749-956052ea-645c-41e8-aa70-27254b8e750b.png)

1. Make directories from home directory /home/home02/$USER to save a Singularity docker (see instructions under 0.).
  ```Python
  $cd
  $mkdir /nobackup
  $mkdir /nobackup/$USER
  $touch .singularity
  $ln -s /nobackup/$USER/ .singularity [did not work but could be ignored?]
  $mkdir /nobackup/$USER/firedrake
  
```
2.	Load modules required for installation. Can be done from home directory or Firedrake directory; type "module avail" to check names of mvapich or other modules (it had changed to name stated below):
```Python
/nobackup/$USER/firedrake
```


```Python
 $ module swap openmpi 
 $ module swap openmpi mvapich2/2.3.1
 $ module add apptainer
 $ module add anaconda
```
       
3.	Install Firedrake using Singularity:

```Python
  $ cd /nobackup/$USER/firedrake
  $  apptainer pull docker://firedrakeproject/firedrake
  ```
4.	When finishing the installation, a Firedrake image named firedrake_latest.sif is made.
After that, move the Firedrake image into a directory where you want to compute. From directory where Firedrake has been installed, i.e.:

```Python
$cd /nobackup/$USER/firedrake
```

```Python
   $mv firedrake_latest.sif /home/home02/$USER
```

5.	Now, the installation ends. To run a Firedrake code (from where: the home directory where .sif file is), there are two ways. 

(a)	Command directly at a linux window. First, load Singularity as 
```Python
        $module add singularity/3.6.4
```
  
  And then, use the following command (after adding a test file, here named as a "BL_test.py" file and making a .cache dirctory in the home directory):
 
```Python
mkdir .cache
$singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache firedrake_latest.sif python BL_test.py
```

(b) Save in nobackup:
save_path =  "/nobackup/$USER/lin_pot_flow/"


(c) Command for checking Firedrake:
```Python
$singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache firedrake_latest.sif python BL_test.py

(b)	Submit a job script to job scheduler of ARC4. An example of a job script is attached in file "example.sh", to submit type (from home directory --see https://arcdocs.leeds.ac.uk/usage/batchjob.html):
```Python
qsub example.sh

```

(d)	Submit a job script to job scheduler of ARC4. An example of a job script is attached in file "example.pdf".  The command to submit a job script is
```Python
$qsub example.sh
```
For more information, refer to https://arcdocs.leeds.ac.uk/usage/batchjob.html.


## Visualisation

Paraview

Download data on arc4 from local machine, e.g (change to your USER):
scp amtjch@arc4.leeds.ac.uk:/home/home02/helm.py .



