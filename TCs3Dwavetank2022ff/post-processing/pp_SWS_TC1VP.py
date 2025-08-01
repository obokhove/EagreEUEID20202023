# Post-processing code for TC1 using the MMP-VP approach and the GLL-SV old approach
# Validate the numerical results against the standing wave solutions of linear potential-flow eqs at one point

import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os.path

#------ User Input ------#
scheme ='MMP' # MMP or SV
save_figure = False

Lx = 2.0*np.pi
res_x=0.05
Nx = round(Lx/res_x) 
dx = Lx/Nx 

Aa = 0.01
Bb = 0.01
m1 = 2
k1 = 2*np.pi*m1/Lx
g  = 9.81
kH0 = np.arccosh(0.5*g)
H0 = kH0/k1
w = math.sqrt(2*k1*np.sinh(kH0))
Tw = 2*np.pi/w
#------------------------#
# Absolute adata_path (change to your settings), e.g.:
adata_path='/Users/amtob/werkvuurdraak2025/Wavetank2022-2025-main/'

if scheme=='MMP':
	# old: data_path1='TC_MMP/data/MMP/TC1_dt' # simulation using dt
	# old: data_path2='TC_MMP/data/MMP/TC1_dt_half' # simulation using dt/2
        # new:
        data_path1=adata_path +'data/MMP/TC1_dt' # simulation using dt
        data_path2=adata_path +'data/MMP/TC1_dt_half' # simulation using dt/2
        figure_name ='TC0_MMP.png'
else: # scheme=='SV'
	# old: data_path1='TC_SV/data/SV/TC1_dt'
	# old data_path2='TC_SV/data/SV/TC1_dt_half'
        # new:
        data_path1=adata_path +'data/SV/TC1_dt'
        data_path2=adata_path +'data/SV/TC1_dt_half'
        figure_name ='TC0_SV(GLL).png'

if save_figure:
    path=os.path.join(data_path1,'figures')
    try: 
        os.makedirs(path, exist_ok = True) 
    except OSError as error: 
        print(error)
    save_path = os.path.join(path, figure_name)

def load_binary(file_path, file_name):
    """load data from binary files (.npy) at x=0"""
    file = os.path.join(file_path, file_name)
    with open(file,'rb') as f:
        data_array = np.load(f)
    h   = data_array[0,1]
    if scheme=='MMP':
        phis = data_array[0,3]
        phib = data_array[0,4]
    else:
        phis = data_array[0,2]
        phib = data_array[0,3]
    return h, phis, phib

# get t and energy from the energy file
energy_file1 = os.path.join(data_path1,'energy.csv')
with open(energy_file1,'r') as ef1:
	if scheme=='MMP':
		t1, E_tot1 = np.loadtxt(ef1, usecols=(0,3), unpack=True)
	else:
		t1, E_tot1 = np.loadtxt(ef1, usecols=(0,1), unpack=True)
en01 = np.array([(E_tot1[i]-E_tot1[0]) for i in range(len(E_tot1))])

energy_file2 = os.path.join(data_path2,'energy.csv')
with open(energy_file2,'r') as ef2:
	if scheme=='MMP':
		t2, E_tot2 = np.loadtxt(ef2, usecols=(0,3), unpack=True)
	else:
		t2, E_tot2 = np.loadtxt(ef2, usecols=(0,1), unpack=True)
en02 = np.array([(E_tot2[i]-E_tot2[0]) for i in range(len(E_tot2))])

# numerical results
h1=[]
phis1=[]
phib1=[]

for t in t1:
    tt   = format(t,'.4f')
    fname = tt+'.npy'
    h_n, phis_n, phib_n = load_binary(data_path1, fname)
    h1.append(h_n)
    phis1.append(phis_n)
    phib1.append(phib_n)

# exact solutions
he = H0 + np.cos(k1*0) * (Aa*np.cos(w*t1) + Bb*np.sin(w*t1))
phise = np.cos(k1*0) * 2 * np.cosh(k1*H0) * (-Aa*np.sin(w*t1) + Bb*np.cos(w*t1))/w
phibe = np.cos(k1*0) * 2 * (-Aa*np.sin(w*t1) + Bb*np.cos(w*t1))/w  

# plot
title_size=16

plt.figure(num=1, figsize=(7,9),constrained_layout=True)
ax1 = plt.subplot(311)
ax2 = plt.subplot(312)
ax3 = plt.subplot(313)

ax1.set_xlabel('Time [s]',fontsize=12)
ax1.set_ylabel('$E(t)-E(t_0)$ [J]',fontsize=12)
ax1.plot(t1,en01,  'b-', label=r'$E(\Delta t)$')
ax1.plot(t2,en02,  'c-', label=r'$E(\Delta t/2)$')
ax1.plot(t2,4*en02,'r--',label=r'$4E(\Delta t/2)$')
if scheme=='MMP':
	ax1.set_title('Energy variations (modified-midpoint scheme)', fontsize=title_size)
else:
	ax1.set_title('Energy variations (Störmer-Verlet scheme)', fontsize=title_size)
ax1.legend(loc='upper right')
ax1.set_xticks(np.arange(0, 3*Tw+0.01, Tw))
ax1.set_xticklabels(['0', r'$T_p$', r'2$T_p$', r'3$T_p$'])
ax1.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
ax1.grid()

ax2.set_title(r'$h(x=0,t)$', fontsize=title_size)
ax2.set_xlabel('Time [s]',fontsize=12)
ax2.set_ylabel(r'$h$ [m]',fontsize=12)
ax2.plot(t1,he,'b-',label='"Exact Linear"')
ax2.plot(t1,h1,'r--',label='Numerical')
#ax2.plot(t2,h2,'g-.',label='Numerical(dt/2)')
ax2.legend(loc='upper right')
ax2.set_xticks(np.arange(0, 3*Tw+0.01, Tw))
ax2.set_xticklabels(['0', r'$T_p$', r'2$T_p$', r'3$T_p$'])
ax2.grid()

ax3.set_title(r'$\phi(x=0,z,t)$', fontsize=title_size)
ax3.set_ylabel(r'$\phi$',fontsize=12)
ax3.set_xlabel('Time [s]',fontsize=12)
ax3.plot(t1,phise,'b-',label='"Exact linear", $z=H_0$')
ax3.plot(t1,phis1,'r--',label='Numerical, $z=H_0$')
ax3.plot(t1,phibe,'g-',label='"Exact linear", $z=0$')
ax3.plot(t1,phib1,'y--',label='Numerical, $z=0$')
ax3.legend(loc='upper left')
ax3.set_xticks(np.arange(0, 3*Tw+0.01, Tw))
ax3.set_xticklabels(['0', r'$T_p$', r'2$T_p$', r'3$T_p$'])
ax3.grid()

if save_figure:
    plt.savefig(save_path,dpi=300)
else:
    plt.show()
