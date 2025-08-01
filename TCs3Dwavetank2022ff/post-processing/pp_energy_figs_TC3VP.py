import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#------ User Input ------#
# Absolute adata_path (change to your settings), e.g.:
adata_path='/Users/amtob/werkvuurdraak2025/Wavetank2022-2025-main/'

# Old
#file_MMP1='data_VP/MMP/TC3_3D_dt1_velocity/checkpoints.csv' # dt=0.001s
#file_MMP2='data_VP/MMP/TC3_3D_dt2_velocity/checkpoints.csv' # dt=0.002s
#file_SV1='data_Re/SV/TC3_3D_dt1_velocity/checkpoints.csv'
#file_SV2='data_Re/SV/TC3_3D_dt2_velocity/checkpoints.csv'
# New can be run in post-processing folder:
file_MMP2 = adata_path +'data/MMP/TC3_dt1_test/checkpoints.csv' # dt=0.001s
file_MMP1 = adata_path +'data/MMP/TC3_dt2_test/checkpoints.csv' # dt=0.002s
file_SV2 = adata_path +'data/SV/TC3_dt1_test/checkpoints.csv'  # dt=0.001s
file_SV1 = adata_path +'data/SV/TC3_dt2_test/checkpoints.csv'  # dt=0.002s

label1='E($\Delta t_1$)'
label2='E(2$\Delta t_1$)'

save_figure=True
figure_name_1='TC3_energy_1.png'
figure_name_2='TC3_energy_2.png'

t_stop=5.670 # when the R_t=0 for the first time
#------------------------#

ts_row_number=round(t_stop/0.002)
print(ts_row_number)
ts=ts_row_number-1

with open(file_MMP1,'r') as MMP1:
    t1_MMP, E1_tot_MMP = np.loadtxt(MMP1, usecols=(0,1), unpack=True)

with open(file_MMP2,'r') as MMP2:
    t2_MMP, E2_tot_MMP = np.loadtxt(MMP2, usecols=(0,1), unpack=True)

with open(file_SV1,'r') as SV1:
    t1_SV, E1_tot_SV = np.loadtxt(SV1, usecols=(0,1), unpack=True)

with open(file_SV2,'r') as SV2:
    t2_SV, E2_tot_SV = np.loadtxt(SV2, usecols=(0,1), unpack=True)

en1_MMP = np.array([(E1_tot_MMP[i]-E1_tot_MMP[0]) for i in range(len(E1_tot_MMP))])
en2_MMP = np.array([(E2_tot_MMP[i]-E2_tot_MMP[0]) for i in range(len(E2_tot_MMP))])

en1_SV = np.array([(E1_tot_SV[i]-E1_tot_SV[0]) for i in range(len(E1_tot_SV))])
en2_SV = np.array([(E2_tot_SV[i]-E2_tot_SV[0]) for i in range(len(E2_tot_SV))])

plt.figure(num=1, figsize=(9,8),constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.set_title('(a) Energy variations with the modified-midpoint scheme', fontsize=16)
ax1.set_xlabel('Time [s]',fontsize=14)
ax1.set_ylabel('E(t) [J]',fontsize=14)
ax1.plot(t1_MMP,en1_MMP,'r-',label='MMP1: '+label1)
ax1.plot(t2_MMP,en2_MMP,'b--',label='MMP2: '+label2)
ax1.legend(loc='best',fontsize=14)
ax1.grid()

ax2.set_title('(b) Energy variations with the Störmer-Verlet scheme', fontsize=16)
ax2.set_xlabel('Time [s]',fontsize=14)
ax2.set_ylabel('E(t) [J]',fontsize=14)
ax2.plot(t1_SV,en1_SV,'r-',label='SV1: '+label1)
ax2.plot(t2_SV,en2_SV,'b--',label='SV2: '+label2)
ax2.legend(loc='best',fontsize=14)
ax2.grid()

if save_figure:
    plt.savefig(figure_name_1,dpi=300)

# in the absence of wavemaker motion

plt.figure(num=2, figsize=(9,8),constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.set_title('(a) Energy variations with the modified-midpoint scheme ($t>t_{\mathrm{stop}}$)', fontsize=16)
ax1.set_xlabel('Time [s]',fontsize=14)
ax1.set_ylabel('$E(t)-E(t_{\mathrm{stop}})$ [J]',fontsize=14)
ax1.plot(t1_MMP[ts+1:],en1_MMP[ts+1:]-en1_MMP[ts+1],'c-',label=label1)
ax1.plot(t2_MMP[ts+1:],en2_MMP[ts+1:]-en2_MMP[ts+1],'b-',label=label2)
ax1.plot(t1_MMP[ts+1:],4*(en1_MMP[ts+1:]-en1_MMP[ts+1]),'r--',label='4'+label1)
ax1.set_xlim(t1_MMP[ts],t1_MMP[-1])
ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax1.legend(loc='upper right',fontsize=14,framealpha=0.9)
ax1.grid()

ax2.set_title('(b) Energy variations with the Störmer-Verlet scheme ($t>t_{\mathrm{stop}}$)', fontsize=16)
ax2.set_xlabel('Time [s]',fontsize=14)
ax2.set_ylabel('$E(t)-E(t_{\mathrm{stop}})$ [J]',fontsize=14)
ax2.plot(t1_SV[ts:],en1_SV[ts:]-en1_SV[ts],'c-',label=label1)
ax2.plot(t2_SV[ts:],en2_SV[ts:]-en2_SV[ts],'b-',label=label2)
ax2.plot(t1_SV[ts:],4*(en1_SV[ts:]-en1_SV[ts]),'r--',label='4'+label1)
ax2.set_xlim(t1_SV[ts],t1_SV[-1])
ax2.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax2.legend(loc='upper right',fontsize=14,framealpha=0.9)
ax2.grid()
print('MMP:', t1_MMP[ts+1], t2_MMP[ts+1])
print(' SV:', t1_SV[ts],    t2_SV[ts])
if save_figure:
    plt.savefig(figure_name_2,dpi=300)
else:
    plt.show()
