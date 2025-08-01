# Compute the rate of convergence using Atiken Extrapolation for TC1
# data source: psi and h

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import os.path

# -----------------------
nCG  = 'CG1'
N_t0 = 0 # the ordinal corresponding to the start time for averaging
N_t1 = -1

# coarse mesh
#data_path1 = 'data_VP/MMP/TC1_2D_CG1_series(1ele,phiGLL)/dx0128_nz16_4dtf'
# medium 
#data_path2 = 'data_VP/MMP/TC1_2D_CG1_series(1ele,phiGLL)/dx0064_nz16_4dtf'
# fine mesh
#data_path3 = 'data_VP/MMP/TC1_2D_CG1_series(1ele,phiGLL)/dx0032_nz16_4dtf'

# Absolute adata_path (change to your settings), e.g.:
adata_path='/Users/amtob/werkvuurdraak2025/Wavetank2022-2025-main/'

# coarse mesh
# Old: data_path1 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx200_nz32_2dtf'
# New (untested, run in post-processing directory)
data_path1 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx200_nz32_2dtf'
# medium 
# Old: data_path2 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx400_nz32_2dtf'
data_path2 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx400_nz32_2dtf'
# fine mesh
# Old: data_path3 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz32_2dtf'
data_path3 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz32_2dtf'

save_figure = False

figure_name1 = 's_psi(dtf8-4-2).png'
figure_name2 = 's_h(dtf8-4-2).png'

colour1 = 'blue' # L2 norms
colour2 = 'red'  # L^inf norms
# -----------------------

if save_figure:
    # Old: path ='data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/'
    # New: 
    path = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/'
    save_path_1 = os.path.join(path, figure_name1)
    save_path_2 = os.path.join(path, figure_name2)

def load_data(file_path, file_name):
	"""load data from a binary file"""
	binary_file = os.path.join(file_path, file_name)
	with open(binary_file,'rb') as bf:
		data_array = np.load(bf)
	h   = data_array[:,0]
	psi = data_array[:,1]
	return h, psi

# get t from the energy file
energy_file = os.path.join(data_path1,'energy.csv')
with open(energy_file,'r') as e_f:
    t_steps_file = np.loadtxt(e_f, usecols=0)
t_steps=t_steps_file[:-1]

print('Averaging from ', t_steps[N_t0])
print('Averaging to ', t_steps[N_t1])

# rate of convergence 's' based on h
s_L2_numerator = []
s_L2_denominator = []
s_Linf_numerator = []
s_Linf_denominator = []

# rate of convergence 's' based on psi
s_L2_numerator1 = []
s_L2_denominator1 = []
s_Linf_numerator1 = []
s_Linf_denominator1 = []

for t in t_steps:
	# numerical results
	tt   = format(t,'.4f')
	fname = tt+'.npy'

	h_c, psi_c = load_data(data_path1,fname)
	h_m, psi_m = load_data(data_path2,fname)
	h_f, psi_f = load_data(data_path3,fname)

	# based on h
	s_L2_numerator.append(norm(h_m-h_c))   
	s_L2_denominator.append(norm(h_f-h_m)) 
	s_Linf_numerator.append(np.max(np.abs(h_m-h_c)))
	s_Linf_denominator.append(np.max(np.abs(h_f-h_m)))

	# based on psi
	s_L2_numerator1.append(norm(psi_m-psi_c))   
	s_L2_denominator1.append(norm(psi_f-psi_m)) 
	s_Linf_numerator1.append(np.max(np.abs(psi_m-psi_c)))
	s_Linf_denominator1.append(np.max(np.abs(psi_f-psi_m)))

# Aitken Extrapolation (psi)
s_L2_num1 = np.array(s_L2_numerator1)
s_L2_den1 = np.array(s_L2_denominator1)
s_Li_num1 = np.array(s_Linf_numerator1)
s_Li_den1 = np.array(s_Linf_denominator1)

s_L2_psi = np.log2(s_L2_num1/s_L2_den1)
s_Li_psi = np.log2(s_Li_num1/s_Li_den1)

ave_s_L2_psi = np.average(s_L2_psi[N_t0:N_t1])
ave_s_Li_psi = np.average(s_Li_psi[N_t0:N_t1])

plt.figure(num=1, figsize=(10,9),constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.set_title(r'Approximated $L^2$- and $L^{\infty}$-norms between $\{\psi_c, \psi_m, \psi_f\}$', fontsize=20)
ax1.set_xlabel('$t$ [s]',fontsize=14)
ax1.set_ylabel('$L^2$-norm',color=colour1, fontsize=14)
ax1.ticklabel_format(style='scientific', scilimits=(0,0))
ax1.tick_params(axis='y', labelcolor=colour1)
ax1.plot(t_steps, s_L2_num1, 'b-',  label=r'$||\psi_m-\psi_c||_{2}$')
ax1.plot(t_steps, s_L2_den1, 'b--', label=r'$||\psi_f-\psi_m||_{2}$')
ax1.plot(t_steps, 4*s_L2_den1,'g-.',label=r'$4||\psi_f-\psi_m||_{2}$')

ax1r = ax1.twinx()
ax1r.set_ylabel(r'$L^\infty$-norm',color=colour2, fontsize=14)
ax1r.ticklabel_format(style='scientific', scilimits=(0,0))
ax1r.tick_params(axis='y', labelcolor=colour2)
ax1r.plot(t_steps, s_Li_num1,  'r-', label=r'$||\psi_m-\psi_c||_{\infty}$')
ax1r.plot(t_steps, s_Li_den1, 'r--', label=r'$||\psi_f-\psi_m||_{\infty}$')
ax1r.plot(t_steps, 4*s_Li_den1, 'y-.', label=r'$4||\psi_f-\psi_m||_{\infty}$')

ax1.legend(loc='upper left',fontsize=14)
ax1r.legend(loc='upper right',fontsize=14)
ax1.set_xlim(0,t_steps[-1])
ax1.grid()

ax2.set_title('Rate of Convergence ('+nCG+')', fontsize=20)
ax2.set_xlabel('$t$ [s]',fontsize=14)
ax2.set_ylabel('$s(t)$',fontsize=14)
ax2.plot(t_steps, s_L2_psi,  'b-', label=r'$L^2, \bar{s}=$'+str(ave_s_L2_psi))
ax2.plot(t_steps, s_Li_psi, 'r--', label=r'$L^{\infty}, \bar{s}=$'+str(ave_s_Li_psi))
ax2.legend(loc='best',fontsize=14)
ax2.set_xlim(0,t_steps[-1])
ax2.grid()

if save_figure:
    plt.savefig(save_path_1, dpi=300)

# Aitken Extrapolation (h)
s_L2_num = np.array(s_L2_numerator)
s_L2_den = np.array(s_L2_denominator)
s_Li_num = np.array(s_Linf_numerator)
s_Li_den = np.array(s_Linf_denominator)

s_L2 = np.log2(s_L2_num/s_L2_den)
s_Li = np.log2(s_Li_num/s_Li_den)

ave_s_L2 = np.average(s_L2[N_t0:N_t1])
ave_s_Li = np.average(s_Li[N_t0:N_t1])

plt.figure(num=2, figsize=(10,9),constrained_layout=True)
ax1 = plt.subplot(211)
ax2 = plt.subplot(212)

ax1.set_title(r'Approximated $L^2$- and $L^{\infty}$-norms between $\{h_c, h_m, h_f\}$', fontsize=20)
ax1.set_xlabel('$t$ [s]',fontsize=14)
ax1.set_ylabel('$L^2$-norm',color=colour1, fontsize=14)
ax1.ticklabel_format(style='scientific', scilimits=(0,0))
ax1.tick_params(axis='y', labelcolor=colour1)
ax1.plot(t_steps, s_L2_num, 'b-',  label=r'$||h_m-h_c||_{2}$')
ax1.plot(t_steps, s_L2_den, 'b--', label=r'$||h_f-h_m||_{2}$')
ax1.plot(t_steps, 4*s_L2_den,'g-.',label=r'$4||h_f-h_m||_{2}$')

ax1r = ax1.twinx()
ax1r.set_ylabel(r'$L^\infty$-norm',color=colour2, fontsize=14)
ax1r.ticklabel_format(style='scientific', scilimits=(0,0))
ax1r.tick_params(axis='y', labelcolor=colour2)
ax1r.plot(t_steps, s_Li_num,  'r-', label=r'$||h_m-h_c||_{\infty}$')
ax1r.plot(t_steps, s_Li_den, 'r--', label=r'$||h_f-h_m||_{\infty}$')
ax1r.plot(t_steps, 4*s_Li_den, 'y-.', label=r'$4||h_f-h_m||_{\infty}$')

ax1.legend(loc='upper left',fontsize=14)
ax1r.legend(loc='upper right',fontsize=14)
ax1.set_xlim(0,t_steps[-1])
ax1.grid()

ax2.set_title('Rate of Convergence ('+nCG+')', fontsize=20)
ax2.set_xlabel('$t$ [s]',fontsize=14)
ax2.set_ylabel('$s(t)$',fontsize=14)
ax2.plot(t_steps, s_L2,  'b-', label=r'$L^2, \bar{s}=$'+str(ave_s_L2))
ax2.plot(t_steps, s_Li, 'r--', label=r'$L^{\infty}, \bar{s}=$'+str(ave_s_Li))
ax2.legend(loc='best',fontsize=14)
ax2.set_xlim(0,t_steps[-1])
ax2.grid()

if save_figure:
    plt.savefig(save_path_2, dpi=300)
else:
	plt.show()
