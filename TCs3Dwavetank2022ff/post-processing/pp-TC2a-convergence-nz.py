
import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import os.path

# -------------------------------------------------------------------------
m = 2
Lx = 2*np.pi
k = 2*np.pi*m/Lx
g  = 9.81
kH0 = np.arccosh(0.5*g)
H0 = kH0/k
dtf=(Lx/3200)/np.pi/2 # finest time step

# Absolute adata_path (change to your settings), e.g.:
adata_path='/Users/amtob/werkvuurdraak2025/Wavetank2022-2025-main/'

# Old:
#data_path = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx3200_nz64_dtf'
#data_path1 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz32_2dtf'
#data_path2 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz16_2dtf'
#data_path3 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz8_2dtf'
#data_path4 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz6_2dtf'
#data_path5 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz5_2dtf'
#data_path6 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz4_2dtf'
#data_path7 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz3_2dtf'
#data_path8 = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz2_2dtf'

# New (untested, run in post-processing directory):
data_path = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx3200_nz64_dtf'
data_path1 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz32_2dtf'
data_path2 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz16_2dtf'
data_path3 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz8_2dtf'
data_path4 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz6_2dtf'
data_path5 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz5_2dtf'
data_path6 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz4_2dtf'
data_path7 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz3_2dtf'
data_path8 = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/Nx800_nz2_2dtf'

#dz = np.array([H0/32, H0/16, H0/8, H0/4, H0/3, H0/2])
nz = np.array([32,16,8,6,5,4,3,2])
Nz = nz+1

one_point = True # only plot the results at t=Tend
results = 'h' #'h/psi'
save_figure = True
figure_name = 'error-nz-semilog('+results+').png'
# -------------------------------------------------------------------------

if save_figure:
    # Old: path = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)'
    # New:
    path = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)'
    save_path = os.path.join(path, figure_name)

def load_binary(file_path, file_name, results):
    """load data from binary files (.npy)"""
    file = os.path.join(file_path, file_name)
    with open(file,'rb') as f:
        data_array = np.load(f)
    if results == 'h':
        return data_array[:,0]
    elif results == 'psi':
    	return data_array[:,1]

def Li_norm(array1,array2):
	"""compute the L_infity norm between two numpy arrays"""
	return np.max(np.abs(array1-array2))

def L2_norm(array1,array2):
	"""compute the L_2 norm between two numpy arrays"""
	diff = (array1-array2)**2
	return np.sqrt(np.sum(diff))

# get t from the energy file
energy_file = os.path.join(data_path1,'energy.csv')
with open(energy_file,'r') as e_f:
    t_steps_full = np.loadtxt(e_f, usecols=0)

if one_point:
	t_plot = [4.3] # change
	plt.figure(num=1)
else:
	t_plot = t_steps_full[0:-1:30] # change
	plt.figure(num=1,figsize=(9,7))

for t in t_plot:
	tt   = format(t,'.4f')
	fname = tt+'.npy'
	u = load_binary(data_path,fname,results)
	u1 = load_binary(data_path1,fname,results)
	u2 = load_binary(data_path2,fname,results)
	u3 = load_binary(data_path3,fname,results)
	u4 = load_binary(data_path4,fname,results)
	u5 = load_binary(data_path5,fname,results)
	u6 = load_binary(data_path6,fname,results)
	u7 = load_binary(data_path7,fname,results)
	u8 = load_binary(data_path8,fname,results)

	L2 = np.array([norm(u1-u), norm(u2-u), norm(u3-u), norm(u4-u), norm(u5-u), norm(u6-u), norm(u7-u), norm(u8-u)])
	#L2 = np.array([L2_norm(h1,h), L2_norm(h2,h), L2_norm(h3,h), L2_norm(h4,h), L2_norm(h5,h)])
	Linf = np.array([Li_norm(u1,u), Li_norm(u2,u), Li_norm(u3,u), Li_norm(u4,u),\
		             Li_norm(u5,u), Li_norm(u6,u), Li_norm(u7,u), Li_norm(u8,u)])
	
	if one_point:
		col = 'blue'
		plt.semilogy(nz,L2,  color=col, marker='^',label=r'$L^2$')
	else:
		col = (np.random.random(), np.random.random(), np.random.random()) #random color
		plt.semilogy(nz,L2,  color=col, marker='^',label=r'$L^2$'+f' ($t={t:.4f}$)')
	
	plt.semilogy(nz,Linf,color=col, marker='o',ls='--',label=r'$L^{\infty}$')
	#plt.loglog(dz,L2,  color=col, marker='^',label=r'$L^2$'+f' (t={t:.3f})')
	#plt.loglog(dz,Linf,color=col, marker='o',ls='--',label=r'$L^{\infty}$')


#ref = [ele**2 for ele in dz]
#plt.loglog(dz,ref,'r-.',lw='2',label='$(\Delta z)^2$')

if results == 'h':
	plt.title(r'Spatial convergence analysis based on $h(x_i,t^n)$', fontsize=16)
elif results == 'psi':
	plt.title(r'Spatial convergence analysis based on $\psi(x_i,t^n)$', fontsize=16)
#plt.xlabel(r'$\Delta z$',fontsize=14)
plt.xlabel(r'$n_z$',fontsize=14)
plt.ylabel(r'$\mathcal{E}$',fontsize=14)
if one_point:
	plt.legend(ncols=1, fontsize=14)
else:
	plt.legend(ncols=1, fontsize=10)
plt.grid()
plt.tight_layout()

if save_figure:
    plt.savefig(save_path, dpi=300)
else:
    plt.show()
