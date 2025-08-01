# Numerical Wavetank in *Firedrake*: Two Method Implementations
This repository shares the source code for a potential-flow-based numerical wavetank (NWT) built upon Luke's variational principle (VP). Two numerical models have been established and implemented in [*Firedrake*](https://www.firedrakeproject.org/): 
- **Method 1**: Weak formulations are manually derived from the space-discretised VP and explicitly formulated in the code.
- **Method 2**: Weak formulations are generated automatically from the encoded time-discretised VP, resulting in an implicit implementation.

The NWT has been verified and validated through a series of test cases (TCs). Their source codes are shared to support reproducibility and encourage further development.

## 🗂️ Code Files for Test Cases 1–4
- **Method 1**:
    - `3D_tank.py`: main script;
    - `settings_TCx.py`: test case configuration file;
    - `savings.py`: simulation output management;
    - `solvers_full.py`: explicit weak formulations for the SE and SV schemes.
- **Method 2**:
    - `3D_tank_VP.py`: main script;
    - `settings_TCx.py`: test case configuration file;
    - `savings.py`: simulation output management.
- :new: In both methods, the time variables in wavemaker-related functions have been replaced with *Firedrake* `Constant` objects to improve computational performance.
- The folder [202002](202002) contains experimental data for TC4, including sampled wavemaker input and probe measurements.


## :wrench: Simulation Instructions
1. **Select a test case**
    - In the main script (`3D_tank.py` or `3D_tank_VP.py`), set the test case by changing: `case = 'TCx'`.
2. **Set simulation parameters**
    - Configure simulation-specific parameters in the corresponding `settings_TCx.py` file:
      - Set the time-stepping scheme via `scheme`: `SE`/`SV` for Method 1, `MMP` for Method 2;
      - Define the output directory by modifying the `save_path` variable;
      - Adjust spatial resolution via `res_x`, `res_y` and `nz` in the `domain` function; set temporal resolution and output frequency via `dt` and `dt_save` in the `set_time` function.
    - Key parameters that must be adjusted for each test case are summarised below. Note: `save_path` must be set individually for each run.

| Test Case | Parameters to configure in `settings_TCx.py` |
| :---:     |    :----    |
| TC1       |  Two simulations for both MMP and SV (should be run in serial): <br/> 1. `dt = min(dx,dz)/pi`, `dt_save = dt` in function `set_time`; <br/> 2. `dt = 0.5*(min(dx,dz)/pi)`, `dt_save = 2*dt`. |
| TC2a      |  Method 2 (MMP), serial or parallel.  <br/> Finest resolution for the reference solution: `res_x = 2*pi/3200`, `n_z = 64`, `dt = (Lx/3200)/(2*pi)`, `dt_save = dt*64`. <br/> Five computations with different horizontal resolutions: `res_x = 2*pi/{50/100/200/400/800}`, where `n_z = 32`, `dt = ((Lx/3200)/(2*pi))*2`, `dt_save = dt*32` remain unchanged. <br/> Eight computations with different vertical resolutions: `n_z = {2/3/4/5/6/8/16/32}`, where `res_x = 2*pi/800`, `dt = ((Lx/3200)/(2*pi))*2`, `dt_save = dt*32` remain unchanged.|
| TC2b      |  Six computations with different temporal resolutions: `dt = ((Lx/3200)/(2*pi))*{2/4/8/16/32/64}` (`dt_save = dt*{32/16/8/4/2/1}`), where `res_x = 2*pi/800`, `n_z = 32` remain unchanged. |
| TC2c      | Thirteen computations needed. The coarsest resolution for $u_{crc}$ is choosen as `res_x = 2*pi/50`, `n_z = 4`, `dt = ((Lx/3200)/(2*pi))*64`, `dt_save = dt`. The other two finer vertical resolutions are chosen as `n_z = 6` and `n_z = 8`. |
| TC3       | Two simulations for MMP and SV (serial or parallel): <br/> 1. `dt = 0.001` in function `set_time`; <br/> 2. `dt = 0.002` (`dt_save = 0.002` remains unchanged). |
| TC4       | One simulation with both methods (serial or parallel): switch between `scheme = "SV"` and `scheme = "MMP"`, 'Tend=120', 't_save=0.01'. |

3. **Output files**
    - After the simulation completes, output files will be saved to the specified directory. The expected files vary by test case: 
      - TC1/TC2: `energy.csv` containing time series of energy in the fluid domain; a series of `.npy` binary files containing 1D field data named after the time step, and `readme.txt`, which is a summary of the simulation details;
      - TC3: `checkpoints.csv` containing energy, water depths at three vertices and wavemaker-related data, `readme.txt`;
      - TC4: `energy.csv`, `probes.csv` containing numerical measurements, `readme.txt`.

## :bar_chart: Reproducing the Results
The numerical solutions can be processed and visualised using the post-processing scripts provided in the folder [post-processing](post-processing). The table below summarises the scripts used to reproduce each figure or table.

:warning: **Note**: Before running a script, make sure to adjust the `data_path` variable (and any relevant output paths) to match your simulation output directory.

| Test Case| Figure/Table No. | Data Sources | Post-Processing Script  | Notes |
|  :----:  |  :----:    |    :----:    |      :----:           | :--- |
|   TC1    |   Fig.3    |  `energy.csv`, <br/> `.npy` files | `pp_SWS_TC1VP.py` | Set `scheme`, and `data_path{1/2}` for both methods |
|   TC2a   |   Fig.4(a),(b)  |  `.npy` files | `pp-TC2a-convergence-dx.py` | Set `data_path{/1/2/3/4/5}`, and `path` in `save_figure`. |
|   TC2a   |   Fig.4(c),(d)  |  `.npy` files | `pp-TC2a-convergence-nz.py` | Set `data_path{/1/2/3/4/5/6/7/8}`, and `path` in `save_figure`. |
|   TC2b   |   Fig.4(e),(f)  |  `.npy` files | `pp-TC2b-convergence-dt.py` | Set `data_path{/1/2/3/4/5/6}`, and `path` in `save_figure`. |
|   TC2c   |   Table 1, 2    |  `.npy` files | `pp-TC2c-Atiken.py` | Set `data_path{1/2/3}` for a group computations with coarse/medium/fine meshes. The time-averaged order of convergence is shown in the bottom figure. |
|   TC2    |   Fig.5, Fig.6  |  `.npy` files |  `pp-TC2-advanced-convergence.py` | Set `data_folder` and thirteen `data_path_xxx`; switch `results` between `'h'/'psi'`. |
|   TC3    |   Fig.7, Fig.8  |  `checkpoints.csv` | `pp_energy_figs_TC3VP.py` | Set the four directories `file_MMP{1/2}` and `file_SV{1/2}`. |
|   TC4    |   Fig.10    |  `probes.csv` <br/> folder `202002` | `pp_wavemaker_TC4VP.py`  | Set the directories of the wavemaker data `file_exp{1/2}` and numerical results `data_path`. |
|   TC4    |   Fig.12   |  `probes.csv` <br/> folder `202002` | `pp_probes_TC4VP.py`  | Set the measurements directory via `measurements`, and the output directories of the two methods via `data_path` and `data_path1` respectively. |
|   TC4    |   Fig.13   |  `probes.csv` <br/> folder `202002` | `FFT_202002.m`  | Set the probe number using `probe = '{1/2/3/4/5/6}'`, the scheme-specific directory for `probes.csv` in `Data_num`, and the directory of Elena's results in `Data_num2`. |
| TC5       |**`pot_sp2.py`,`PFE_sp2max.py`,** <br/> `PFE_sp2maxA.py`, `PFE_sp2A.py`, `PFE_sp2energy.py`<br/> in the folder TC5_codes <br/> :white_check_mark: Done. 20240724


## Notes on test cases TC1, TC2, TC3, TC4 (OB all done via Docker installation on M2 Macbook)
OB reproducing, testing and plotting 28-07-2025 to 01-08-2025. Draft of manuscript for figure reference added.

TC4 used MPI 10 cores 28-07-2025 to 30-07-2025:
- datapath set in settings_TC4.py (should be set right)
- set dt_save = 0.02 and Tend = 120 (0.1 for a short test)
- New file FFT_202002.m (only works from main directory not yet from post-processing)
- SV 10 cores on M2 Macbook 2024: 17:11:47.1hr
- MMP 10 cores on M2 Macbook 2024: 35:01:50.5hr
- all figures reproduced
- All post-processing files have all been heavily adapted/corrected and made more user-friendly: pp_wavemaker_TC4VP.py , pp_probes_TC4VP.py , FFT_202002.m

TC3 used MPI 10 cores 30-07-2025 to 01-08-2025:
- made changes in, new files: settings_TC3.py, pp_energy_figs_TC3VP.py 
- Started 3D_tank.py on 10 processes dt=0.002: 1:37:25.5hr
- Started 3D_tank.py on 10 processes dt=0.001: 3:06:17.2hr
- Started 3D_tank_VP.py on 10 processes dt = 0.001: 8:50:1.6hr
- Started 3D_tank_VP.py on 10 processes dt = 0.002: 4:25:57.5hr
- Figs. 7, 8: reproduced.

TC2 untested but adapted post-processing files.

TC1 (serial) 01-08-2025:
- files adapated
- SV-dt: 38.6s
- SV-dt/2: 48.6s
- MMP-dt: 40.4s
- MMP-dt/2: 57.4s

TC5: not tested again.




