Describe singularity. Virtual environment. HPC provide module singularity. Where does on get this.

How to install Firedrake in Singularity on ARC4.
1. Make directory a directory for Singularity at "nobackup" and $USER.  
  $mkdir  /nobackup/cscmaw
  $mkdir  /nobackup/cscmaw/$USER
  $touch  .singularity
  $ln -s /nobackup/cscmaw/$USER/  .singularity
  $mkdir /nobackup/cscmaw/firedrake
  
 2. Load modules.
  $module swap mvapich
  $module add apptainer
  $module add anaconda
  
3. Install Firedrake at "firedrake" using Singularity.
   $cd /nobackup/cscmaw/firedrake
   $apptainer pull docker://firedrakeproject/firedrake
   
4. After finishing installation of Firedrake, move the firedrake image, "firedrake_latest.sif" to $USER.
   $mv firedrake_latest.sif /home/home02/$USER
 
5. To run firedrake, there are two ways:
    (a) Type the follows directlty at a linux window:
       $mpirun -n 1 singularity exec --env 'PATH=/home/firedrake/firedrake/bin:$PATH' -B /run -B /nobackup -B ~/.cache:/home/firedrake/firedrake/.cache firedrake_latest.sif python helm.py
     
    (b) Submit a job script to HPC job scheduler. An example "exaple.pdf" is attached.
        (ref:https://arcdocs.leeds.ac.uk/usage/batchjob.html)
            

   [example.pdf](https://github.com/obokhove/EagreEUEID20202023/files/11069032/example.pdf)


  
