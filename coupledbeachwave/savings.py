# savings file for the 2D coupled wavetank
# Main file: "coupled_tank.py"

from firedrake import *
import os.path
import numpy as np


""" *******************************************************
    *            Compute the energy difference            *
    ******************************************************* """

# Updated: Add B, C and b term in A

def dw_energy(Lw, H0, g, H, h_n0, psi_1_n0, hat_psi_n0, 
              Ww, IoW, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,  
              A11, AN1, ANN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, I1, IN):

    energy = assemble((  (0.5*Lw*Lw*IoW*h_n0) * ((psi_1_n0.dx(0)**2)*M11 \
                                                        + dot(hat_psi_n0.dx(0), (2.0*MN1*psi_1_n0.dx(0)+dot(MNN,hat_psi_n0.dx(0))))) \
                           -(Lw*Lw*IoW*h_n0.dx(0)) * ( psi_1_n0.dx(0) * (D11*psi_1_n0 + dot(D1N,hat_psi_n0))\
                                                      +dot(hat_psi_n0.dx(0),(DN1*psi_1_n0+dot(DNN, hat_psi_n0))) )\
                           -(Lw*Lw*IoW*H0*b.dx(0)) * ( psi_1_n0.dx(0) * (B11*psi_1_n0 + dot(B1N,hat_psi_n0))\
                                                      +dot(hat_psi_n0.dx(0),(BN1*psi_1_n0+dot(BNN, hat_psi_n0))) )\
                           + (0.5*Lw*Lw*IoW*(h_n0.dx(0)**2)/h_n0) * (S11*psi_1_n0**2 \
                                                                     + dot(hat_psi_n0, (2.0*SN1*psi_1_n0 + dot(SNN,hat_psi_n0))))\
                           + (0.5*H0*H0/h_n0) * (FWF*IoW*Lw*Lw*(b.dx(0)**2) + Ww) * (A11*psi_1_n0**2 \
                                                                     + dot(hat_psi_n0, (2.0*AN1*psi_1_n0 + dot(ANN,hat_psi_n0))))\
                           + ( Lw*Lw*IoW*H0*b.dx(0)*h_n0.dx(0)/h_n0 ) * (C11*psi_1_n0**2 \
                                                                     + dot(hat_psi_n0, (2.0*CN1*psi_1_n0 + dot(CNN,hat_psi_n0))))\
                           + 0.5*H0*g*Ww*(h_n0 - H)**2 )*dx)

    return energy


# separated from sw_solver

def sw_energy(Nvol, d_x, g, Hr, h_fv, hu_fv):
    U = np.zeros((2,Nvol+1))
    U[0,1:Nvol+1] = h_fv.dat.data[:]
    U[1,1:Nvol+1] = hu_fv.dat.data[:]

    # only for the internal cells
    hu_square = np.zeros((1, Nvol-1))
    h_square  = np.zeros((1, Nvol-1))
    hH_square = np.zeros((1, Nvol-1))

    huu = np.zeros((1, Nvol+1))
    Ind = np.where(U[0,:] >= 1e-9)
    huu[0,Ind] = U[1,Ind]**2/U[0,Ind] # huu = (hu)^2/h = hu^2
    hu_square[0,:] = huu[0, 1:Nvol]
    hH_square[0, :] = (U[0, 1:Nvol] - Hr[0:-1])**2 # (h-Hr)^2
    
    E_sw = d_x*(0.5*np.sum(hu_square[0,:]) + 0.5*g*np.sum(hH_square[0,:]))

    return E_sw

#------------------------------------------------------------------------#
#                           3D solution (x,y,z)                          #
#------------------------------------------------------------------------#
def dw_phi_projection(i, n_z, waves, Indz, psi_s, psi_i):
    if i==n_z:                                      
        waves.dat.data[Indz[i]] = psi_s.dat.data[:]
    else:                                                    
        waves.dat.data[Indz[i]] = psi_i.dat.data[:,n_z-1-i]
# yl notes: assembling phi, including the surface and the interior values
# yl Q: how is data storaged in psi_i (based on a Vector Function Space)?
# A in 2025: it is consistent with the maths derivation/modelling.

#-------------------------------------------------------------------------------------#
#                                 Transform the domain                                #
#-------------------------------------------------------------------------------------#

#__________________________________ z-transform (2D) _________________________________#
def dw_z_transform(mesh_2D, n_z, h_1D, beach_1D, H0, Indz):
    for i in range(0, n_z+1):                                          # for each layer
        mesh_2D.coordinates.dat.data[Indz[i],1]*=h_1D.dat.data[:]/H0   # z -> z*h/H0
        mesh_2D.coordinates.dat.data[Indz[i],1]+=beach_1D.dat.data[:]  # z -> z+b(x)

#_____________________________ x-transform (2D, rewrite) _____________________________#
def dw_x_transform(mesh_2D, Lw, WM_2D):
    mesh_2D.coordinates.dat.data[:,0] *= (Lw-WM_2D.dat.data[:])/Lw
    mesh_2D.coordinates.dat.data[:,0] += WM_2D.dat.data[:]

def save_README(README_file, L_total, Ldw, Lsw, H0, xb, sb, Ne_dw, Nv_sw, n_z, gamma, Tw, w, t_stop, Lw, 
                dt, t, jours, heures, minutes, secondes, comp_time, before_it, process):
    README_file.write('======================================\n')
    README_file.write('                Summary               \n')
    README_file.write('======================================\n\n')

    README_file.write('------ Dimensions of the domain ------\n')
    README_file.write('Total length  : %10.4f m\n' %(L_total))
    README_file.write('Coupling point: %10.4f m\n' %(Ldw))
    README_file.write('Depth  H0     : %10.4f m\n' %(H0))
    README_file.write('Beach start at:  %5s m\n' %str(xb))
    README_file.write('Beach slope   :  %5s\n\n' %str(sb))

    README_file.write('----------- Mesh resolutions ---------\n')
    README_file.write('DW: %10.5f m (%-5s elements)\n' % (Ldw/Ne_dw,str(Ne_dw)))
    README_file.write('SW: %10.5f m (%-5s volumns)\n'  % (Lsw/Nv_sw,str(Nv_sw)))
    
    README_file.write('In z: %5s layers\n\n' % (str(n_z+1)))

    README_file.write('-------------- Wavemaker -------------\n')
    README_file.write('Amplitude: %10.4f m\n' %(gamma))
    README_file.write('Period:    %10.4f s\n' %(Tw))
    README_file.write('Frequency: %10.4f s-1\n' %(w))
    README_file.write('Stops after %6.2f periods (%8.4f s)\n' %(t_stop/Tw, t_stop))
    README_file.write('Lw:         %5s m\n\n' %(str(Lw)))

    README_file.write('------------- Final time -------------\n')
    README_file.write('Tend: %10.4f s\n' %(t))
    README_file.write('Δt:   %12.8f s\n\n' %(dt))
    
    README_file.write('Simulation was run on %2s process(es).\n' %(str(process)))
    README_file.write('Before time loop: %4.2f s\n' %(before_it))
    README_file.write('Computation time: %-2s d %-2s h %-2s min %4.2f s\n' %(str(jours),str(heures),str(minutes), secondes))
    README_file.write('That is in total: %10.2f s\n' %(comp_time))