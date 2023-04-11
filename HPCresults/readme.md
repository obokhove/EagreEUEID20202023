## Instruction to install Firedrake in a Singularity docker on HPC-ARC4

( OB et al. To do: Describe singularity. Virtual environment. HPC provides module singularity.
Onno 09-04-2023: from which directory & machine are these commands given?
Poor editting as well, which I have started to fix. Steps 2 and 3 do not work (sigh):
```Python
[amtob@login1.arc4 ~]$ mkdir /nobackup/cscmaw
mkdir: cannot create directory ‘/nobackup/cscmaw’: File exists
[amtob@login1.arc4 ~]$ 
[amtob@login1.arc4 ~]$ mkdir  /nobackup/cscmaw/amtob
mkdir: cannot create directory ‘/nobackup/cscmaw/amtob’: Permission denied
[amtob@login1.arc4 ~]$ 
[amtob@login2.arc4 ~]$ ln -s /nobackup/amtob/  .singularity
ln: failed to create symbolic link ‘.singularity’: File exists
```
)

( Via JB from Firedrake; there is a guide to using Singularity/Apptainer using the Docker image as a starting point here: https://github.com/firedrakeproject/firedrake/wiki/singularity![image](https://user-images.githubusercontent.com/16267535/230949453-6f2c3661-a080-46ff-8779-704679c56ea6.png) 

Reply Junho (not tested yet), there are two directories:
```Python
/nobackup/$USER/firedrake
/home/home02/$USER
```

The former is where Singularity docker for Firedrake is installed, the latter is where you run Firedrake. So when you install Singularity, go to the former, type the commands at the former directory. To run Firedrake, go to the latter directory. 


)


Junho's instructions to date 09-04-2023:
0. [OB] Log onto the ARC4 HPC (?), see: https://arcdocs.leeds.ac.uk/getting_started/logon/logon-off-campus.html#connecting-from-linux-macos-systems

To run larger jobs use queu:
https://arcdocs.leeds.ac.uk/usage/batchjob.html![image](https://user-images.githubusercontent.com/16267535/230985749-956052ea-645c-41e8-aa70-27254b8e750b.png)


1. Onno's 1. Presumably it is:
  ```Python
  $cd
  $mkdir  /nobackup/$USER
  $touch  .singularity
  $ln -s /nobackup/$USER/  .singularity
  $ mkdir /nobackup/$USER/firedrake
  
  
```

1.	JUNHO 1. Make directories [OB: presumably on ARC4] to save a Singularity docker (see instructions under 0.).
  ```Python
  $cd
  $mkdir  /nobackup/cscmaw
  $mkdir  /nobackup/cscmaw/$USER
  $touch  .singularity
  $ln -s /nobackup/cscmaw/$USER/  .singularity
  $ mkdir /nobackup/cscmaw/firedrake
  
```
2.	Load modules required for installation. ( ONNO: unclear: from which directory are commands typed? )
```Python
 $ module swap openmpi mvapich
 $ module add apptainer
 $ module add anaconda
```
       
3.	Install Firedrake using Singularity ( ONNO: unclear: from which directory are commands typed? )
```Python
  $ cd /nobackup/cscmaw/firedrake
  $  apptainer pull docker://firedrakeproject/firedrake
  ```
4.	When finishing the installation, a Firedrake image named firedrake_latest.sif is made.
After that, move the Firedrake image into a directory where you want to compute.

( ONNO: unclear: from which directory are commands typed? )

```Python
   $mv firedrake_latest.sif /home/home02/$USER
```

5.	Now, the installation ends. To run a Firedrake code, there are two ways.

( ONNO: unclear: from which directory are commands typed? )

(a)	Command directly at a linux window. First, load Singularity as 
```Python
        $ module add singularity/3.6.4
```
  
  And then, use the following command:
  
```Python
$singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache firedrake_latest.sif python BL_test.py
```

(b)	Submit a job script to job scheduler of ARC4. An example of a job script is attached:
```Python
[OB Add script]
```

