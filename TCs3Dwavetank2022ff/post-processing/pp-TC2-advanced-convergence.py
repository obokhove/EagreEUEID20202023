
# nz: r<s<v
# s-r = v-s

import numpy as np
from numpy.linalg import norm
from matplotlib import pyplot as plt
import os.path

# -------------------------------------------------------------------------
Lx = 2*np.pi
dtf=(Lx/3200)/(2*np.pi)
x_coor = np.linspace(0, Lx, 50+1) # coarsest mesh verticies

# Absolute adata_path (change to your settings), e.g.:
adata_path='/Users/amtob/werkvuurdraak2025/Wavetank2022-2025-main/'

# old
#data_folder = 'data_VP/MMP/TC1_2D_series2(1ele,phiGLL)/'

# New (untested, run in post-processing directory):
data_folder = adata_path+'data/MMP/TC1_2D_series2(1ele,phiGLL)/'
data_path_crc = data_folder+'c4c_Nx50_nz4_64dtf'
data_path_mrc = data_folder+'m4c_Nx100_nz4_64dtf'
data_path_frc = data_folder+'f4c_Nx200_nz4_64dtf'
data_path_crm = data_folder+'c4m_Nx50_nz4_32dtf'
data_path_crf = data_folder+'c4f_Nx50_nz4_16dtf'
data_path_csc = data_folder+'c6c_Nx50_nz6_64dtf'
data_path_msc = data_folder+'m6c_Nx100_nz6_64dtf'
data_path_fsc = data_folder+'f6c_Nx200_nz6_64dtf'
data_path_csm = data_folder+'c6m_Nx50_nz6_32dtf'
data_path_csf = data_folder+'c6f_Nx50_nz6_16dtf'
data_path_cvc = data_folder+'c8c_Nx50_nz8_64dtf'
data_path_mvc = data_folder+'m8c_Nx100_nz8_64dtf'
data_path_cvm = data_folder+'c8m_Nx50_nz8_32dtf'

data_path_ref = data_folder+'Nx3200_nz64_dtf'
dx = np.array([Lx/50, Lx/100, Lx/200])

t_plot = 4.3

results = 'psi' #'h/psi'
save_figure = True
figure_name1 = 'Crs_and_Crt('+results+').png'
figure_name2 = 'Cst_im_ex('+results+').png'
figure_name3 = 'C_exponential('+results+').png'
# -------------------------------------------------------------------------

if save_figure:
    save_path1 = os.path.join(data_folder, figure_name1)
    save_path2 = os.path.join(data_folder, figure_name2)
    save_path3 = os.path.join(data_folder, figure_name3)

def load_binary(file_path, file_name, results):
    """load data from binary files (.npy)"""
    file = os.path.join(file_path, file_name)
    with open(file,'rb') as f:
        data_array = np.load(f)
    if results == 'h':
        h_f = data_array[:,0]
        return h_f[::64] # coarsest mesh verticies
    elif results == 'psi':
    	psi_f = data_array[:,1]
    	return psi_f[::64]


tt   = format(t_plot,'.4f')
fname = tt+'.npy'
u_crc = load_binary(data_path_crc,fname,results) #numpy.ndarray
u_mrc = load_binary(data_path_mrc,fname,results)
u_frc = load_binary(data_path_frc,fname,results)
u_crm = load_binary(data_path_crm,fname,results)
u_crf = load_binary(data_path_crf,fname,results)

u_csc = load_binary(data_path_csc,fname,results)
u_msc = load_binary(data_path_msc,fname,results)
u_fsc = load_binary(data_path_fsc,fname,results)
u_csm = load_binary(data_path_csm,fname,results)
u_csf = load_binary(data_path_csf,fname,results)

u_cvc = load_binary(data_path_cvc,fname,results)
u_mvc = load_binary(data_path_mvc,fname,results)
u_cvm = load_binary(data_path_cvm,fname,results)
	
u_ref = load_binary(data_path_ref,fname,results) # finest mesh
	
Rp = (u_crc-u_mrc)/(u_mrc-u_frc)
Rq = (u_crc-u_crm)/(u_crm-u_crf)

Cr_s= Rp*(u_mrc-u_msc)/(Rp-1) + Rq*(u_crm-u_csm)/(Rq-1) - (Rp*Rq-1)*(u_crc-u_csc)/((Rp-1)*(Rq-1)) # Cr-Cs
Cr_v= Rp*(u_mrc-u_mvc)/(Rp-1) + Rq*(u_crm-u_cvm)/(Rq-1) - (Rp*Rq-1)*(u_crc-u_cvc)/((Rp-1)*(Rq-1)) # Cr-Ct

Rps = (u_csc-u_msc)/(u_msc-u_fsc)
Rqs = (u_csc-u_csm)/(u_csm-u_csf)
Cs_v= Rps*(u_msc-u_mvc)/(Rps-1) + Rqs*(u_csm-u_cvm)/(Rqs-1) - (Rps*Rqs-1)*(u_csc-u_cvc)/((Rps-1)*(Rqs-1))
	
#u0 = Rp*u_msc/(Rp-1) + Rq*u_csm/(Rq-1) - (Rp*Rq-1)*u_csc/((Rp-1)*(Rq-1))
#u0 = Rp*u_mtc/(Rp-1) + Rq*u_ctm/(Rq-1) - (Rp*Rq-1)*u_ctc/((Rp-1)*(Rq-1))

#col = 'blue'
#col = (np.random.random(), np.random.random(), np.random.random()) #random color
#plt.loglog(dx,L2,  color=col, marker='^',label=r'$L^2$'+f' (t={t:.4f})')
#plt.loglog(dx,Linf,color=col, marker='o',ls='--',label=r'$L^{\infty}$')
plt.figure(num=1,figsize=(6,4))
plt.plot(x_coor, Cr_s, 'r-', label='$C_r-C_s$')
plt.plot(x_coor, Cr_v, 'b--', label='$C_r-C_v$')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('Error',fontsize=14)
plt.xlim(0, Lx)
plt.grid()
plt.legend(loc='upper right',ncols=1, fontsize=14)
plt.tight_layout()
if save_figure:
    plt.savefig(save_path1, dpi=300)

plt.figure(num=2,figsize=(6,4))
plt.plot(x_coor,Cs_v, 'g-', label='$C_s-C_v$')
plt.plot(x_coor,Cr_v-Cr_s, 'k--', label='$(C_r-C_v)-(C_r-C_s)$')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('Error',fontsize=14)
plt.xlim(0, Lx)
plt.grid()
plt.legend(loc='upper right',ncols=1, fontsize=14)
plt.tight_layout()
#print(Li_norm(u_crc,u_csc),Li_norm(u_mrc,u_msc),Li_norm(u_crm,u_csm))
#print(Li_norm(u_csc,u_ctc),Li_norm(u_msc,u_mtc),Li_norm(u_csm,u_ctm))

if save_figure:
    plt.savefig(save_path2, dpi=300)

# point-wise evaluation of the base number C
C1 = np.power(abs((u_csc-u_cvc)/(u_crc-u_csc)),1/2)
C2 = np.power(abs((u_msc-u_mvc)/(u_mrc-u_msc)),1/2)
C3 = np.power(abs((u_csm-u_cvm)/(u_crm-u_csm)),1/2)

plt.figure(num=3,figsize=(6,4))
plt.plot(x_coor,C1, 'b-', label=r'$\{u_{crc},u_{csc},u_{cvc}\}$')
plt.plot(x_coor,C2, 'r-', label=r'$\{u_{mrc},u_{msc},u_{mvc}\}$')
plt.plot(x_coor,C3, 'g-', label=r'$\{u_{crm},u_{csm},u_{cvm}\}$')
plt.xlabel('$x$',fontsize=14)
plt.ylabel('$C$',fontsize=14)
plt.xlim(0, Lx)
plt.grid()
plt.legend(loc='upper right',ncols=1, fontsize=14)
plt.tight_layout()
if save_figure:
    plt.savefig(save_path3, dpi=300)
else:
    plt.show()




