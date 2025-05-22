# FV solver for the nonlinear shallow water equations
# Originally created on Mon Nov 23 14:28:55 2015 by FG
# Modified by YL in March 2025

# k:   0     1     2     3                 Nv-1   Nv   <-- Cell index
#   |-----|-----|-----|-----|......|-----|-----|-----|
#      ^ x_c                                ^     ^  L
# ^: ghost cells

# Notes by YL:
# - ? it seems there are rwo ghost cells set at the right end, which is unnecessary
# - ? What BC shoud be applied at the right side of the SW region for hu?
# - ! The original expression for the potential energy was wrong
# - Although h and hu are updated separately, two-row vectors are needed throughout as 
#   both of them are required when calculating the wave speed S_R/L. 

import numpy as np
from firedrake import *

def U_half(h, U):
    # extracting velocity u
    u = np.zeros_like(U[0, :])  # Shape: (Nvol-2,)
    Ind = np.where(U[0,:]!=0)
    u[Ind]=U[1,Ind]/U[0,Ind]
    
    # calculate the vector U_{k+-1/2}^{+-}, see 6.22, 6.23 of FG's thesis
    u_half = np.zeros((2, len(U[0, :])))  # Shape: (2, Nvol-2)
    u_half[0,:]=h
    u_half[1,:]=h*u
    return [u,u_half]

def flux_half(U_half, g):
    F = np.zeros((2,len(U_half[0,:])))
    F[0,:] = U_half[1,:]
    Ind = np.where(U_half[0,:]!=0)
    F[1, Ind] = (U_half[1, Ind]**2)/U_half[0, Ind] + 0.5*g*U_half[0, Ind]**2  
    return F

def right_wave_speed(ul, ur, hl, hr, g):
    Sr = np.maximum(ul+ np.sqrt(g*hl), ur+ np.sqrt(g*hr))
    return Sr

def left_wave_speed(ul, ur, hl, hr, g):
    Sl = np.minimum(ul- np.sqrt(g*hl), ur- np.sqrt(g*hr))
    return Sl

def HLL_flux(Fl, Fr, Sl, Sr, Ul, Ur):
    F = np.zeros((2,len(Sl)))

    Ind = np.where(Sl>0)
    F[:, Ind] = Fl[:, Ind] # use left flux

    Ind = np.where( (Sl<=0) & (0<=Sr) & (Sl!=Sr) )
    F[:,Ind] = (Sr[Ind]*Fl[:,Ind] - Sl[Ind]*Fr[:,Ind] + Sl[Ind]*Sr[Ind]*(Ur[:,Ind]-Ul[:,Ind]))/(Sr[Ind]-Sl[Ind])

    Ind = np.where(Sr<0)
    F[:,Ind] = Fr[:,Ind] # use right flux
    return F
    
def solve_FV(Nvol, d_x, dt, bk, bk_half_r, bk_half_l, g, h_bc, hu_bc, h_fv, hu_fv, hu_fe):

    #---------- Solutions U^n: ----------#
    U = np.zeros((2,Nvol+1)) # two rows, including ghost cells
    U[0,1:Nvol+1] = h_fv.dat.data[:]
    U[1,1:Nvol+1] = hu_fv.dat.data[:]
    U[1,-1] = -U[1,-2] # !!! hu
    U[0,-1] = U[0,-2]
    
    #---- Boundary conditions at x_c ----#
    U[0,0] = h_bc
    U[1,0] = hu_bc
    
    #-------- Initialise U^{n+1} --------#
    U_next = np.copy(U) # create a NEW array
    
    ############################################################################
    #                                 Update h                                 #
    ############################################################################            
    Uk = np.copy(U[:, 1:Nvol])          # Internal cells (1 to Nvol-1)
    Uk_plus  = np.copy(U[:, 2:Nvol+1])  # U_{k+1}
    Uk_minus = np.copy(U[:, 0:Nvol-1])  # U_{k-1}
    
    #--------------------------- Non-negative depth ---------------------------#
    h_plus_r  = np.maximum(Uk_plus[0,:] + bk[0, 2:Nvol+1] - bk_half_r, 0)   # h_{k+1/2^+}
    h_plus_l  = np.maximum(Uk[0,:]      + bk[0, 1:Nvol]   - bk_half_r, 0)   # h_{k+1/2^-}
    h_minus_r = np.maximum(Uk[0,:]      + bk[0, 1:Nvol]   - bk_half_l, 0)   # h_{k-1/2^+}
    h_minus_l = np.maximum(Uk_minus[0,:]+ bk[0, 0:Nvol-1] - bk_half_l, 0)   # h_{k-1/2^-}

    #----- Left and right values of U at the interface k+-1/2 -----#
    [u_plus_l,U_plus_l] = U_half(h_plus_l, Uk)            # U(k+1/2)-
    [u_plus_r,U_plus_r] = U_half(h_plus_r, Uk_plus)       # U(k+1/2)+
    [u_minus_l,U_minus_l] = U_half(h_minus_l, Uk_minus)   # U(k-1/2)-
    [u_minus_r,U_minus_r] = U_half(h_minus_r, Uk)         # U(k-1/2)+
    
    # Left and right values of flux at the interface k+-1/2 #
    Fl_plus  = flux_half(U_plus_l,g)   # F(U_{k+1/2}-)
    Fr_plus  = flux_half(U_plus_r,g)   # F(U_{k+1/2}+)
    Fl_minus = flux_half(U_minus_l,g)  # F(U_{k-1/2}-)
    Fr_minus = flux_half(U_minus_r,g)  # F(U_{k-1/2}+)
    
    #--------------------- Left and right speeds at each interface of cell k ----------------------#
    Sl_plus = left_wave_speed(u_plus_l,u_plus_r,h_plus_l,h_plus_r,g)      # Ul/r = U(k+1/2)-/+
    Sr_plus = right_wave_speed(u_plus_l,u_plus_r,h_plus_l,h_plus_r,g)     # Ul/r = U(k+1/2)-/+
    Sl_minus = left_wave_speed(u_minus_l,u_minus_r,h_minus_l,h_minus_r,g) # Ul/r = U(k-1/2)-/+
    Sr_minus = right_wave_speed(u_minus_l,u_minus_r,h_minus_l,h_minus_r,g)# Ul/r = U(k-1/2)-/+
    
    #---------------------------------- HLL fluxes ----------------------------------#
    F_minus = HLL_flux(Fl_minus, Fr_minus,  Sl_minus, Sr_minus, U_minus_l, U_minus_r )
    F_plus  = HLL_flux(Fl_plus, Fr_plus, Sl_plus, Sr_plus, U_plus_l, U_plus_r)
    
    #-------------------------------- Update h --------------------------------#
    U_next[0, 1:Nvol] = Uk[0,:] - dt*(F_plus[0,:] - F_minus[0,:])/d_x
    
    # Boundary condition
    U_next[0,-1] = U_next[0,-2]

    ############################################################################
    #                                 Update hu                                #
    ############################################################################

    Hk       = np.copy(U_next[0, 1:Nvol])
    Hk_plus  = np.copy(U_next[0, 2:Nvol+1])
    Hk_minus = np.copy(U_next[0, 0:Nvol-1])
        
    #--------------------- Non-negative depth ---------------------#
    h_next_plus_r  = np.maximum(Hk_plus + bk[0, 2:Nvol+1] - bk_half_r, 0)
    h_next_plus_l  = np.maximum(Hk      + bk[0, 1:Nvol]   - bk_half_r, 0)
    h_next_minus_r = np.maximum(Hk      + bk[0, 1:Nvol]   - bk_half_l, 0)
    h_next_minus_l = np.maximum(Hk_minus+ bk[0, 0:Nvol-1] - bk_half_l, 0)

    #-- Left and right values of U at the interface k+1/2 --#
    U_plus_l[0,:] = h_next_plus_l         # h_{k+1/2^-}^{n+1}
    U_plus_r[0,:] = h_next_plus_r         # h_{k+1/2^+}^{n+1}
    U_minus_l[0,:] = h_next_minus_l       # h_{k-1/2^-}^{n+1}
    U_minus_r[0,:] = h_next_minus_r       # h_{k-1/2^+}^{n+1}
    
    #- Left and right values of F at the interface k+/-1/2 -#
    Fl_plus  = flux_half(U_plus_l,g)     # Fl(U_{k+1/2})
    Fr_plus  = flux_half(U_plus_r,g)     # Fr(U_{k+1/2})
    Fl_minus = flux_half(U_minus_l,g)    # Fl(U_{k-1/2})
    Fr_minus = flux_half(U_minus_r,g)    # Fr(U_{k-1/2})
    
    #---------------------------- Left and right speeds at each interface of cell k -----------------------------#
    Sl_plus  =  left_wave_speed(u_plus_l, u_plus_r, h_next_plus_l, h_next_plus_r, g)      # Ul/r = U(k+1/2)-/+
    Sr_plus  = right_wave_speed(u_plus_l, u_plus_r, h_next_plus_l, h_next_plus_r, g)     # Ul/r = U(k+1/2)-/+
    Sl_minus =  left_wave_speed(u_minus_l, u_minus_r, h_next_minus_l, h_next_minus_r, g)  # Ul/r = U(k-1/2)-/+
    Sr_minus = right_wave_speed(u_minus_l, u_minus_r, h_next_minus_l, h_next_minus_r, g) # Ul/r = U(k-1/2)-/+
    
    #---------------------------------- HLL fluxes ----------------------------------#
    F_minus = HLL_flux(Fl_minus, Fr_minus,  Sl_minus, Sr_minus, U_minus_l, U_minus_r )
    F_plus  = HLL_flux(Fl_plus, Fr_plus, Sl_plus, Sr_plus, U_plus_l, U_plus_r)

    #--------------------- Source term ---------------------#
    Sk = np.zeros((2,len(Uk[0,:])))
    Sk[1,:] = 0.5*g*h_next_plus_l**2 - 0.5*g*h_next_minus_r**2
    
    #-------------------------------- Update hu -------------------------------#
    U_next[1, 1:Nvol] = Uk[1,:] - dt*(F_plus[1,:] - F_minus[1,:])/d_x + dt*Sk[1,:]/d_x

    # BC for DW: the flux at the left boundary
    hu_fe.vector().set_local(F_minus[0,0])

    #--------- Boundary conditions --------#
    U_next[1,-1] = -U_next[1,-2] # !!!
    
    #-------- Update the solutions -------#
    h_fv.vector().set_local(U_next[0,1:])
    hu_fv.vector().set_local(U_next[1,1:])

    return h_fv, hu_fv, hu_fe
