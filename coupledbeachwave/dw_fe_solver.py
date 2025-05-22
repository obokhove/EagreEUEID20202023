# 2D weak formulations using SE scheme for the deep-water region.
# symmetric matrices only call XN1 (mind B, D)
# This code is based on "solvers_full.py" from the 3D NWT code, but removing the mild-slope approximation

from firedrake import *

"""
    ************************************************************************************************************************
    *                                  Weak Formulations for the symplectic Euler scheme                                   *
    ************************************************************************************************************************ """
#--------------------------------------------------------------------------------------------------------------------------#
#                         Step 1 : Update h at time t^{n+1} and psi_i at time t^* simultaneously:                          #
#__________________________________________________________________________________________________________________________#

# Updated: Add B, C, b term in A, and coupling BC

def WF_h_SE(n_z, g, H0, Lw, dWM_dt, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_n0, hat_psi_star,
            Xx, Ww, IoW, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF, hu_fe,
            M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):

    WF_h = (H0*delta_psi*(h_n1-h_n0)*Ww/dt \
              -  (   IoW*(Lw*Lw)*h_n1 * (psi_1_n0.dx(0)*M11 + dot(hat_psi_star.dx(0),MN1)) * delta_psi.dx(0)\
                  -( IoW*(Lw*Lw)*h_n1.dx(0) ) * ( delta_psi.dx(0) * (D11*psi_1_n0 + dot(D1N,hat_psi_star)) \
                                                 + delta_psi * (D11*psi_1_n0.dx(0) + dot(hat_psi_star.dx(0),DN1)) )\
                  -( IoW*(Lw*Lw)*H0*b.dx(0) ) * ( delta_psi.dx(0) * (B11*psi_1_n0 + dot(B1N,hat_psi_star)) \
                                                 + delta_psi * (B11*psi_1_n0.dx(0) + dot(hat_psi_star.dx(0),BN1)) )\
                  +( IoW*(Lw*Lw)*(h_n1.dx(0)**2)/h_n1 ) * (S11*psi_1_n0 + dot(hat_psi_star,SN1)) * delta_psi\
                  +(H0*H0/h_n1) * (FWF*IoW*(Lw*Lw)*(b.dx(0)**2) + Ww) * (A11*psi_1_n0 + dot(hat_psi_star,AN1)) * delta_psi\
                  +( IoW*(Lw*Lw)*2*H0*b.dx(0)*h_n1.dx(0)/h_n1 ) * (C11*psi_1_n0 + dot(hat_psi_star,CN1)) * delta_psi\
                  - H0*Xx*dWM_dt*h_n1.dx(0)*delta_psi )   )*dx \
            - (delta_psi*Lw*dWM_dt*h_n1*I1) * ds(1) \
            + (delta_psi*(hu_fe*I1*Lw)) * ds(2) # coupling BC:

    WF_hat_psi_star= - ( h_n1*IoW*(Lw*Lw) * elem_mult(delta_hat_star.dx(0),(MN1*psi_1_n0.dx(0) + dot(MNN,hat_psi_star.dx(0))))\
                        -( IoW*(Lw*Lw)*h_n1.dx(0) ) * (elem_mult(delta_hat_star, (psi_1_n0.dx(0)*D1N + dot(DNN.T,hat_psi_star.dx(0)))) \
                                                     + elem_mult(delta_hat_star.dx(0), (DN1*psi_1_n0 + dot(DNN,hat_psi_star)))) \
                        -( IoW*(Lw*Lw)*H0*b.dx(0) ) * (elem_mult(delta_hat_star, (psi_1_n0.dx(0)*B1N + dot(BNN.T,hat_psi_star.dx(0)))) \
                                                     + elem_mult(delta_hat_star.dx(0), (BN1*psi_1_n0 + dot(BNN,hat_psi_star)))) \
                        +( IoW*(Lw*Lw)*(h_n1.dx(0)**2)/h_n1 ) * elem_mult(delta_hat_star,(SN1*psi_1_n0+ dot(SNN,hat_psi_star))) \
                        +(H0*H0/h_n1) * (FWF*IoW*(Lw*Lw)*(b.dx(0)**2) + Ww) * elem_mult(delta_hat_star,(AN1*psi_1_n0+dot(ANN,hat_psi_star))) \
                        +( IoW*(Lw*Lw)*2*H0*b.dx(0)*h_n1.dx(0)/h_n1 ) * elem_mult(delta_hat_star,(CN1*psi_1_n0+dot(CNN,hat_psi_star))) )
              
    WF_hat_BC1 = -Lw*dWM_dt*h_n1*elem_mult(delta_hat_star,IN)
    
    WF_hat_BC2 =  Lw*hu_fe*elem_mult(delta_hat_star,IN) # coupling BC

    WF_h_psi = WF_h + sum((WF_hat_psi_star[ind])*dx for ind in range(0,n_z)) \
                    + sum((WF_hat_BC1[ind])*ds(1) for ind in range(0,n_z)) \
                    + sum((WF_hat_BC2[ind])*ds(2) for ind in range(0,n_z))
    
    return WF_h_psi


#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 2 : Update psi_1 at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# Updated: Add C, b term in A, and coupling BC

def WF_psi_SE(g, H0, H, Lw, dWM_dt, dt, delta_h, psi_1, psi_1_n0, hat_psi_star, h_n1, 
              Xx, Ww, Ww_n1, IoW, b, C11, CN1, CNN, FWF, hu_fe,
              M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN, G1, GN):

    A_psi_s = (H0*Ww_n1*psi_1*delta_h)*dx
        
    L_psi_s = (H0*Ww*psi_1_n0*delta_h \
                    -dt*((Lw*Lw*0.5*IoW*delta_h) * ( psi_1_n0.dx(0)**2*M11 + \
                                                     dot(hat_psi_star.dx(0), (2.0*MN1*psi_1_n0.dx(0)+dot(MNN,hat_psi_star.dx(0)))))\
                         -(Lw*Lw*IoW*delta_h.dx(0)) * ( psi_1_n0.dx(0)*(D11*psi_1_n0 + dot(D1N,hat_psi_star)) \
                                                       +dot(hat_psi_star.dx(0), (DN1*psi_1_n0 + dot(DNN, hat_psi_star))))\
                         +(Lw*Lw*IoW/h_n1) * ( delta_h.dx(0)*h_n1.dx(0) - 0.5*delta_h*(h_n1.dx(0)**2)/h_n1 ) * ( psi_1_n0**2*S11 \
                                                    + 2.0*dot(hat_psi_star,SN1)*psi_1_n0 + dot(hat_psi_star,dot(SNN,hat_psi_star)) )\
                         -(0.5*delta_h*H0*H0/(h_n1**2)) * (FWF*IoW*(Lw*Lw)*(b.dx(0)**2) + Ww) * ( psi_1_n0**2*A11 \
                                                    + 2.0*dot(hat_psi_star,AN1)*psi_1_n0 + dot(hat_psi_star,dot(ANN,hat_psi_star)) )\
                         +(Lw*Lw*IoW*H0*b.dx(0)/h_n1) * (delta_h.dx(0)- delta_h*h_n1.dx(0)/h_n1) * ( psi_1_n0**2*C11 \
                                                    + 2.0*dot(hat_psi_star,CN1)*psi_1_n0 + dot(hat_psi_star,dot(CNN,hat_psi_star)) )\
                         +H0*g*Ww*delta_h*(h_n1-H) - H0*psi_1_n0*Xx*dWM_dt*delta_h.dx(0)) )*dx \
                - dt*(Lw*dWM_dt*delta_h*(psi_1_n0*I1 + dot(hat_psi_star,IN)))*ds(1) \
                - dt*(Lw*delta_h*(psi_1_n0*G1 + dot(hat_psi_star,GN))*hu_fe/h_n1)*ds(2) # coupling BC

    return A_psi_s, L_psi_s


#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 3 : Update psi_i at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# Updated: Add B, C, b term in A, and coupling BC

def WF_hat_psi_SE(H0, n_z, Lw, dWM_dt, dt, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                  Ww, IoW, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF, hu_fe,
                  M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):
    
    a_hat_psi = ( (Lw*Lw*IoW*h_n0) * elem_mult(delta_hat_psi.dx(0), dot(MNN,hat_psi.dx(0))) \
                 -(Lw*Lw*IoW*h_n0.dx(0)) * (elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(0))) + elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_psi)) ) \
                 -(Lw*Lw*IoW*H0*b.dx(0)) * (elem_mult(delta_hat_psi, dot(BNN.T,hat_psi.dx(0))) + elem_mult(delta_hat_psi.dx(0),dot(BNN,hat_psi)) ) \
                 +(Lw*Lw*IoW)*((h_n0.dx(0)**2)/h_n0) * elem_mult(delta_hat_psi,dot(SNN,hat_psi))\
                 +(H0*H0/h_n0) * (FWF*IoW*(Lw*Lw)*(b.dx(0)**2) + Ww) * elem_mult(delta_hat_psi,dot(ANN,hat_psi)) \
                 +(Lw*Lw*IoW*2*H0*b.dx(0)*h_n0.dx(0)/h_n0) * elem_mult(delta_hat_psi,dot(CNN,hat_psi)) )     
            
    L_hat_psi =-( (Lw*Lw*IoW*h_n0) * elem_mult(delta_hat_psi.dx(0), MN1*psi_1_n0.dx(0)) \
                 -(Lw*Lw*IoW*h_n0.dx(0)) * (elem_mult(delta_hat_psi, D1N*psi_1_n0.dx(0)) + elem_mult(delta_hat_psi.dx(0),DN1*psi_1_n0)) \
                 -(Lw*Lw*IoW*H0*b.dx(0)) * (elem_mult(delta_hat_psi, B1N*psi_1_n0.dx(0)) + elem_mult(delta_hat_psi.dx(0),BN1*psi_1_n0)) \
                 +(Lw*Lw*IoW)*((h_n0.dx(0)**2)/h_n0) * elem_mult(delta_hat_psi,SN1*psi_1_n0)\
                 +(H0*H0/h_n0) * (FWF*IoW*(Lw*Lw)*(b.dx(0)**2) + Ww) * elem_mult(delta_hat_psi,AN1*psi_1_n0)\
                 +(Lw*Lw*IoW*2*H0*b.dx(0)*h_n0.dx(0)/h_n0) * elem_mult(delta_hat_psi,CN1*psi_1_n0) )

    hat_psi_BC1 = -Lw*dWM_dt*h_n0*elem_mult(delta_hat_psi,IN)
    hat_psi_BC2 =  Lw*hu_fe*elem_mult(delta_hat_psi,IN)  # coupling BC

    A_hat = sum((a_hat_psi[ind])*dx for ind in range(0,n_z))
    L_hat =   sum((L_hat_psi[ind])*dx for ind in range(0,n_z)) \
            + sum((hat_psi_BC1[ind])*ds(1) for ind in range(0,n_z)) \
            + sum((hat_psi_BC2[ind])*ds(2) for ind in range(0,n_z))

    return A_hat, L_hat
