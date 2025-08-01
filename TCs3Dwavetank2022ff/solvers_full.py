# All the equations have been checked and/or corrected by YL on 24 Oct 2021.
# SE tested by yl. The results were compared with these obtained from solvers.py. 2 Nov 2021.
# The difference between solversUVW.py and this file is the arguments used in each function. 6 Jan 2022.
# This file contains the full version of all the weak forms (rather than using mild-slope approximation). 11 April 2022.

from firedrake import *

"""
    ************************************************************************************************************************
    *                                  Weak Formulations for the symplectic Euler scheme                                   *
    ************************************************************************************************************************ """
#--------------------------------------------------------------------------------------------------------------------------#
#                         Step 1 : Update h at time t^{n+1} and psi_i at time t^* simulataneously:                         #
#__________________________________________________________________________________________________________________________#

# dim, n_z, g, H, H0, Lw, dt, x_coord: constant
# WM, dWM_dy, dWM_dt: at n time level
# delta_psi, delta_hat_star: test functions [shape(delta_psi)=(), ufl.indexed.Indexed; shape(delta_hat_star)=(8,), ufl.tensors.ListTensor]
# h_n0, psi_1_n0: at n time level, known [shape(psi_1_n0.dx(0))=(), ufl.indexed.Indexed]
# >>> h_n1, hat_psi_star = split(w_n1): unknown [shape(hat_pasi_star)=(n_z,), len(hat_psi_star)=n_z]

# delete: WM, dWM_dy, x_coord, H, (dWM_dt)
# add: b, C_ij, B_ij, FWF
def WF_h_SE(dim, n_z, g, H0, Lw, dWM_dt, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_n0, hat_psi_star,
            Uu, Ww, VoW, IoW, WH, XRt, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
            M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):
    
    if dim == "3D":
        WF_h = (H0*delta_psi*(h_n1-h_n0)*Ww/dt \
                -(VoW*h_n1*(psi_1_n0.dx(0)*M11 + dot(hat_psi_star.dx(0),MN1))*delta_psi.dx(0)\
                  +Ww*h_n1*(psi_1_n0.dx(1)*M11 + dot(hat_psi_star.dx(1),MN1))*delta_psi.dx(1) \
                  +Uu*h_n1*(delta_psi.dx(0)*(M11*psi_1_n0.dx(1) + dot(M1N,hat_psi_star.dx(1)))\
                                             + delta_psi.dx(1)*(M11*psi_1_n0.dx(0) + dot(MN1, hat_psi_star.dx(0))))\
                  -(VoW*h_n1.dx(0) + Uu*h_n1.dx(1))*(delta_psi.dx(0)*(D11*psi_1_n0 + dot(D1N,hat_psi_star)) \
                                                     +delta_psi*(psi_1_n0.dx(0)*D11 + dot(hat_psi_star.dx(0),DN1)))\
                  -( Ww*h_n1.dx(1) + Uu*h_n1.dx(0))*(delta_psi.dx(1)*(D11*psi_1_n0 + dot(D1N,hat_psi_star))\
                                                     +delta_psi*(psi_1_n0.dx(1)*D11 + dot(hat_psi_star.dx(1),DN1)))\
                  - H0*(VoW*b.dx(0) + Uu*b.dx(1))*(delta_psi.dx(0)*(B11*psi_1_n0 + dot(B1N,hat_psi_star)) \
                                                +delta_psi*(psi_1_n0.dx(0)*B11 + dot(hat_psi_star.dx(0),BN1)))\
                  - H0*( Ww*b.dx(1) + Uu*b.dx(0))*(delta_psi.dx(1)*(B11*psi_1_n0 + dot(B1N,hat_psi_star))\
                                                +delta_psi*(psi_1_n0.dx(1)*B11 + dot(hat_psi_star.dx(1),BN1)))\
                  +(delta_psi/h_n1)*(  (psi_1_n0*S11+dot(hat_psi_star,SN1))*(VoW*h_n1.dx(0)**2 + Ww*h_n1.dx(1)**2 + 2.0*Uu*h_n1.dx(0)*h_n1.dx(1))\
                               + H0*H0*(A11*psi_1_n0+dot(AN1,hat_psi_star))*(FWF*(VoW*b.dx(0)**2 + Ww*b.dx(1)**2 + 2.0*Uu*b.dx(0)*b.dx(1)) + Ww)\
                              + 2.0*H0*(C11*psi_1_n0+dot(CN1,hat_psi_star))*\
                                                       (VoW*b.dx(0)*h_n1.dx(0)+Uu*(b.dx(0)*h_n1.dx(1)+b.dx(1)*h_n1.dx(0))+Ww*b.dx(1)*h_n1.dx(1)) )\
                  -delta_psi*H0*XRt*h_n1.dx(0)))*dx \
                    - (delta_psi*Lw*dWM_dt*h_n1*I1)*ds(1) 
        
        # checked by yl. JCP (C.1) full weak form!
        # Implement MSA: FWF=0 and Bij=Cij=0.

        WF_hat_psi_star= ( h_n1 * VoW * elem_mult(delta_hat_star.dx(0),(MN1*psi_1_n0.dx(0)+dot(MNN,hat_psi_star.dx(0))))\
                          + Ww * h_n1 * elem_mult(delta_hat_star.dx(1),(MN1*psi_1_n0.dx(1)+dot(MNN,hat_psi_star.dx(1))))\
                          + Uu * h_n1 *(elem_mult((dot(hat_psi_star.dx(0),MNN)+psi_1_n0.dx(0)*M1N),delta_hat_star.dx(1))\
                                       +elem_mult(delta_hat_star.dx(0),(MN1*psi_1_n0.dx(1)+dot(MNN,hat_psi_star.dx(1)))))\
                          -( VoW*h_n1.dx(0) + Uu*h_n1.dx(1) )*(elem_mult(delta_hat_star,(psi_1_n0.dx(0)*D1N + dot(DNN.T,hat_psi_star.dx(0)))) \
                                                              + elem_mult(delta_hat_star.dx(0),(DN1*psi_1_n0 + dot(DNN,hat_psi_star)))) \
                          -(  Ww*h_n1.dx(1) + Uu*h_n1.dx(0) )*(elem_mult(delta_hat_star,(D1N*psi_1_n0.dx(1) + dot(DNN.T,hat_psi_star.dx(1))))\
                                                              + elem_mult(delta_hat_star.dx(1),(DN1*psi_1_n0 + dot(DNN,hat_psi_star))))\
                          - H0*( VoW*b.dx(0) + Uu*b.dx(1) )*(elem_mult(delta_hat_star,(psi_1_n0.dx(0)*B1N + dot(BNN.T,hat_psi_star.dx(0)))) \
                                                              + elem_mult(delta_hat_star.dx(0),(BN1*psi_1_n0 + dot(BNN,hat_psi_star)))) \
                          - H0*(  Ww*b.dx(1) + Uu*b.dx(0) )*(elem_mult(delta_hat_star,(B1N*psi_1_n0.dx(1) + dot(BNN.T,hat_psi_star.dx(1))))\
                                                              + elem_mult(delta_hat_star.dx(1),(BN1*psi_1_n0 + dot(BNN,hat_psi_star))))\
                          + (1.0/h_n1) *   elem_mult(delta_hat_star,(SN1*psi_1_n0 + dot(SNN,hat_psi_star))) * \
                                                                         (VoW*h_n1.dx(0)**2 + Ww*h_n1.dx(1)**2 +2.0*Uu*h_n1.dx(0)*h_n1.dx(1)) \
                          + (H0*H0/h_n1) * elem_mult(delta_hat_star,(AN1*psi_1_n0 + dot(ANN,hat_psi_star))) * \
                                                                         (FWF*(VoW*b.dx(0)**2 + Ww*b.dx(1)**2 + 2.0*Uu*b.dx(0)*b.dx(1)) + Ww)\
                          + (2.0*H0/h_n1)* elem_mult(delta_hat_star,(CN1*psi_1_n0 + dot(CNN,hat_psi_star))) *\
                                                     (VoW*b.dx(0)*h_n1.dx(0)+Uu*(b.dx(0)*h_n1.dx(1)+b.dx(1)*h_n1.dx(0))+Ww*b.dx(1)*h_n1.dx(1)) ) 
        
        # checked by yl. JCP (C.2) full weak form!
        # Implement MSA: FWF=0 and Bij=Cij=0.

    elif dim == "2D":
        WF_h = (H0*delta_psi*(h_n1-h_n0)*Ww/dt \
                -(h_n1*IoW*(Lw*Lw)*(psi_1_n0.dx(0)*M11 + dot(hat_psi_star.dx(0),MN1))*delta_psi.dx(0)\
                  -( IoW*(Lw*Lw)*h_n1.dx(0))*( delta_psi.dx(0)*(D11*psi_1_n0 + dot(D1N,hat_psi_star)) \
                                                      +delta_psi*(psi_1_n0.dx(0)*D11 + dot(hat_psi_star.dx(0),DN1)))\
                  +(1/h_n1)*(IoW*(Lw*Lw)*(h_n1.dx(0)**2))*(psi_1_n0*S11 + dot(hat_psi_star,SN1))*delta_psi\
                  +(Ww*H0*H0/h_n1)*(psi_1_n0*A11 + dot(hat_psi_star,AN1))*delta_psi -delta_psi*H0*XRt*h_n1.dx(0)))*dx\
                - (delta_psi*Lw*dWM_dt*h_n1*I1)*ds(1)
        
        # checked by yl. Remove terms containing y compared with 3D. Vv = Lw*Lw. MSA

        WF_hat_psi_star= ((h_n1*IoW)*(Lw*Lw)*elem_mult(delta_hat_star.dx(0),(MN1*psi_1_n0.dx(0)+dot(MNN,hat_psi_star.dx(0))))\
                          -((Lw*Lw)*h_n1.dx(0)*IoW)*(elem_mult(delta_hat_star, (psi_1_n0.dx(0)*D1N+ dot(DNN.T,hat_psi_star.dx(0)))) \
                                                         + elem_mult(delta_hat_star.dx(0),(DN1*psi_1_n0+dot(DNN,hat_psi_star)))) \
                          +(1.0/h_n1)*((Lw*Lw)*(h_n1.dx(0)**2)*IoW)*elem_mult(delta_hat_star,(SN1*psi_1_n0+ dot(SNN,hat_psi_star)))\
                          + (Ww*H0*H0/h_n1)*elem_mult(delta_hat_star,(AN1*psi_1_n0+dot(ANN,hat_psi_star))))
        
        # checked by yl. Remove terms containing y compared with 3D. Vv = Lw*Lw. MSA
              
    WF_hat_BC = (Lw*dWM_dt*h_n1*elem_mult(delta_hat_star,IN)) # the last term of (C.2)
    
    WF_h_psi = WF_h + sum((WF_hat_psi_star[ind])*dx for ind in range(0,n_z)) + sum((WF_hat_BC[ind])*ds(1) for ind in range(0,n_z))

    return WF_h_psi



#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 2 : Update psi_1 at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# dim, g, H, H0, Lw, dt, x_coord: constant
# WM, dWM_dy, dWM_dt: at n time level
# WM_n1: at n+1 time level
# delta_h: test functions
# psi_1_n0: at n time level, known
# h_n1, hat_psi_star: obtained from step1, known
# >>> psi_1: trial function

# delete: H,WM,WM_n1,dWM_dy,(dWM_dt),x_coord
# add: b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF

def WF_psi_SE(dim, g, H0, Lw, dWM_dt, dt, delta_h, psi_1, psi_1_n0, hat_psi_star, h_n1, 
              Uu, Ww, Ww_n1, VoW, IoW, WH, XRt, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
              M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):
    
    if dim=="3D":
        A_psi_s = (H0*delta_h*Ww_n1*psi_1)*dx 
        #LHS of (C.3)
        
        L_psi_s = (H0*delta_h*Ww*psi_1_n0 \
                  -dt*( delta_h*( 0.5*VoW*((psi_1_n0.dx(0)**2)*M11 + dot(hat_psi_star.dx(0), (2.0*MN1*psi_1_n0.dx(0)\
                                +dot(MNN,hat_psi_star.dx(0)))))\
                                +0.5*Ww*( (psi_1_n0.dx(1)**2)*M11 + dot(hat_psi_star.dx(1),(2.0*MN1*psi_1_n0.dx(1) + dot(MNN,hat_psi_star.dx(1)))) )\
                                +Uu*( psi_1_n0.dx(0)*(M11*psi_1_n0.dx(1) + dot(MN1,hat_psi_star.dx(1)))\
                                + dot(hat_psi_star.dx(0), (MN1*psi_1_n0.dx(1) + dot(MNN,hat_psi_star.dx(1))))))\
                       -(VoW*delta_h.dx(0) + Uu*delta_h.dx(1))*( psi_1_n0.dx(0)*(D11*psi_1_n0 + dot(D1N,hat_psi_star)) \
                                +dot(hat_psi_star.dx(0), (DN1*psi_1_n0 + dot(DNN, hat_psi_star))))\
                       -( Ww*delta_h.dx(1) + Uu*delta_h.dx(0))*( psi_1_n0.dx(1)*(D11*psi_1_n0 + dot(D1N,hat_psi_star))\
                                +dot(hat_psi_star.dx(1), (DN1*psi_1_n0 + dot(DNN, hat_psi_star))))\
                       +(1.0/h_n1)* ( delta_h.dx(0)*(VoW*h_n1.dx(0)+Uu*h_n1.dx(1)) + delta_h.dx(1)*(Ww*h_n1.dx(1)+Uu*h_n1.dx(0))
                                -(delta_h/(2.0*h_n1)) * (VoW*h_n1.dx(0)**2 + Ww*h_n1.dx(1)**2 + 2.0*Uu*h_n1.dx(0)*h_n1.dx(1)) )* \
                             ( psi_1_n0*psi_1_n0*S11 + 2.0*dot(hat_psi_star,SN1)*psi_1_n0 + dot(hat_psi_star,dot(SNN,hat_psi_star)) )\
                       +(H0/h_n1) * ( delta_h.dx(0)*(VoW*b.dx(0)+Uu*b.dx(1)) + delta_h.dx(1)*(Ww*b.dx(1)+Uu*b.dx(0))
                                      -(delta_h/h_n1) * (VoW*h_n1.dx(0)*b.dx(0) + Ww*h_n1.dx(1)*b.dx(1) + Uu*(h_n1.dx(0)*b.dx(1)+h_n1.dx(1)*b.dx(0))) )* \
                                    ( psi_1_n0*psi_1_n0*C11 + 2.0*dot(hat_psi_star,CN1)*psi_1_n0 + dot(hat_psi_star,dot(CNN,hat_psi_star)) )\
   -(0.5*delta_h*H0*H0/(h_n1**2)) * ( psi_1_n0*psi_1_n0*A11 + 2.0*dot(hat_psi_star,AN1)*psi_1_n0 + dot(hat_psi_star,dot(ANN,hat_psi_star)) ) * \
                                    ( FWF*(VoW*b.dx(0)**2 + Ww*b.dx(1)**2 + 2.0*Uu*b.dx(0)*b.dx(1)) + Ww )\
                       +H0*g*Ww*delta_h*h_n1 -H0*g*WH*delta_h - H0*psi_1_n0*XRt*delta_h.dx(0) ))*dx\
                  -dt*(Lw*dWM_dt*delta_h*(psi_1_n0*I1 + dot(hat_psi_star,IN)))*ds(1)
        
        # checked by yl. RHS of (C.3). full weak form!
        # Implement MSA: FWF=0 and Bij=Cij=0.
    
    elif dim=="2D":
        
        A_psi_s = (H0*delta_h*Ww_n1*psi_1)*dx
        
        L_psi_s = -(-H0*delta_h*Ww*psi_1_n0 \
                    +dt*(delta_h*((Lw*Lw)*0.5*IoW)*((psi_1_n0.dx(0)**2)*M11+dot(hat_psi_star.dx(0), (2.0*MN1*psi_1_n0.dx(0)\
                                                                                                           +dot(MNN,hat_psi_star.dx(0)))))\
                         -(IoW*(Lw*Lw)*delta_h.dx(0))*( psi_1_n0.dx(0)*(D11*psi_1_n0 + dot(D1N,hat_psi_star)) \
                                                                 +dot(hat_psi_star.dx(0), (DN1*psi_1_n0 + dot(DNN, hat_psi_star))))\
                         +(1.0/h_n1)*(delta_h.dx(0)*(IoW*h_n1.dx(0)*(Lw*Lw))\
                                    - (delta_h/h_n1)*( (Lw*Lw)*(h_n1.dx(0)**2)*0.5*IoW))*(psi_1_n0*psi_1_n0*S11 \
                                    + 2.0*dot(hat_psi_star,SN1)*psi_1_n0 + dot(hat_psi_star,dot(SNN,hat_psi_star)))\
                         -(0.5*delta_h*Ww*H0*H0/(h_n1**2))*(psi_1_n0*psi_1_n0*A11 + 2.0*dot(hat_psi_star,AN1)*psi_1_n0 \
                                    + dot(hat_psi_star,dot(ANN,hat_psi_star)))\
                         +H0*g*Ww*delta_h*h_n1 -H0*g*WH*delta_h - H0*psi_1_n0*XRt*delta_h.dx(0)))*dx\
                        - dt*(Lw*dWM_dt*delta_h*(psi_1_n0*I1 + dot(hat_psi_star,IN)))*ds(1)
        
        # checked by yl. Remove terms containing y compared with 3D. Vv = Lw*Lw. MSA

    return A_psi_s, L_psi_s


#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 3 : Update psi_i at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# dim, H, H0, n_z, Lw, dt, x_coord: constant
# WM, dWM_dy, dWM_dt: at n time level
# delta_hat_psi: test function
# h_n0, psi_1_n0: at n time level, known
# >>> hat_psi: trial function
# the solution is placed in hat_psi_n0, just for the sake of output.

# delete: H,x_coord,WM, (dWM_dt), dWM_dy

def WF_hat_psi_SE(dim, H0, n_z, Lw, dWM_dt, dt, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                  Uu, Ww, VoW, IoW, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):
    
    if dim=="3D":
        a_hat_psi =(h_n0*VoW*elem_mult(delta_hat_psi.dx(0), dot(MNN,hat_psi.dx(0)))\
                    +Ww*h_n0*elem_mult(delta_hat_psi.dx(1),dot(MNN,hat_psi.dx(1)))\
                    +Uu*h_n0*( elem_mult(dot(hat_psi.dx(0),MNN),delta_hat_psi.dx(1)) + elem_mult(delta_hat_psi.dx(0),dot(MNN,hat_psi.dx(1))) )\
                    -( VoW*h_n0.dx(0) + Uu*h_n0.dx(1))* ( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(0))) \
                                                             + elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_psi)) ) \
                    -(   Ww*h_n0.dx(1) + Uu*h_n0.dx(0)  ) * ( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(1)))\
                                                             + elem_mult(delta_hat_psi.dx(1),dot(DNN,hat_psi)) )\
                    +(1.0/h_n0)*( VoW*(h_n0.dx(0)**2)+Ww*(h_n0.dx(1)**2)+2.0*Uu*h_n0.dx(0)*h_n0.dx(1) )*elem_mult(delta_hat_psi,dot(SNN,hat_psi))\
                    +(Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,dot(ANN,hat_psi)))
        
        L_hat_psi =-(h_n0*VoW*elem_mult(delta_hat_psi.dx(0), MN1*psi_1_n0.dx(0))\
                     +Ww*h_n0*elem_mult(delta_hat_psi.dx(1), MN1*psi_1_n0.dx(1))\
                     +Uu*h_n0*(elem_mult(delta_hat_psi.dx(1), MN1*psi_1_n0.dx(0))\
                                                +elem_mult(delta_hat_psi.dx(0),MN1*psi_1_n0.dx(1)))\
                    -( VoW*h_n0.dx(0) + Uu*h_n0.dx(1))* ( elem_mult(delta_hat_psi, D1N * psi_1_n0.dx(0)) \
                                                            + elem_mult(delta_hat_psi.dx(0),DN1 * psi_1_n0) ) \
                    -(   Ww*h_n0.dx(1) + Uu*h_n0.dx(0)  ) * ( elem_mult(delta_hat_psi, D1N * psi_1_n0.dx(1))\
                                                            + elem_mult(delta_hat_psi.dx(1),DN1 * psi_1_n0) )\
                     +(1.0/h_n0)*( VoW*(h_n0.dx(0)**2)+Ww*(h_n0.dx(1)**2)+2.0*Uu*h_n0.dx(0)*h_n0.dx(1) )*elem_mult(delta_hat_psi,SN1*psi_1_n0)\
                     +(Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,AN1*psi_1_n0))
        
        # MSA
        # checked by yl. Rearrange (C.2): separate _i' (unknown, a) from _1 (known, L) and replace h_n1 with h_n0.
        # ------ questions ------
        # Why use ^n here (instead of ^(n+1) for h and psi_1)? [Because in the solvers section at step 3 we solve for hat_psi_n0.]
        # Why solve for hat_psi_n0 rather than hat_psi_n1? How to obtain hat_psi_n1?

    elif dim=="2D":
        a_hat_psi =(h_n0*(Lw*Lw*IoW)*elem_mult(delta_hat_psi.dx(0),dot(MNN,hat_psi.dx(0)))\
                    -(Lw*Lw*IoW)*h_n0.dx(0)*( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(0)))+elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_psi)) ) \
                    +(1.0/h_n0)*((Lw*Lw)*(h_n0.dx(0)**2)*IoW)*elem_mult(delta_hat_psi,dot(SNN,hat_psi))\
                    +(Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,dot(ANN,hat_psi)))     
            
        L_hat_psi =-((h_n0*IoW)*(Lw*Lw)*elem_mult(delta_hat_psi.dx(0), MN1*psi_1_n0.dx(0))\
                     -((Lw*Lw)*h_n0.dx(0)*IoW)*(elem_mult(delta_hat_psi, D1N*psi_1_n0.dx(0)) \
                                                    + elem_mult(delta_hat_psi.dx(0),DN1*psi_1_n0)) \
                     +(1.0/h_n0)*( (Lw*Lw)*(h_n0.dx(0)**2)*IoW)*elem_mult(delta_hat_psi,SN1*psi_1_n0)\
                     + (Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,AN1*psi_1_n0))
        
        # checked by yl. Remove terms containing y compared with 3D. Vv = Lw*Lw. MSA

    hat_psi_BC = -(Lw*dWM_dt*h_n0*elem_mult(delta_hat_psi,IN))
    A_hat = sum((a_hat_psi[ind])*dx for ind in range(0,n_z))
    L_hat = sum((L_hat_psi[ind])*dx for ind in range(0,n_z)) + sum((hat_psi_BC[ind])*ds(1) for ind in range(0,n_z))

    return A_hat, L_hat

"""
    ************************************************************************************************************************
    *                                   Weak Formulations for the Stormer-Verlet scheme                                    *
    ************************************************************************************************************************ """
#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 1 : Update psi_1^{n+1/2} and psi_i^*:                                    #
#______________________________________________________________________________________________________________________#

# dim, n_z, g, H, H0, Lw, dt, x_coord: constant
# WM, dWM_dy, dWM_dt: at n time level
# WM_half, dWM_half_dy, dWM_half_dt: updated in the time loop, known
# delta_h_sv, delta_hat_psi_sv: test functions (modified by yl.)
# h_n0, psi_1_n0: at n time level, known
# >>> psi_1_half, hat_psi_star = split(w_half): unknown

#delete: x_coord, WM, WM_half, dWM_dy, dWM_dt, dWM_half_dy, (dWM_half_dt), H
# add: b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF

def WF_psi_half_SV(dim, n_z, g, H0, Lw, dt, delta_h_sv, delta_hat_psi_sv, psi_1_n0, psi_1_half, hat_psi_star, h_n0, dWM_half_dt, 
                   Ww, Uu_half, VoW_half, Ww_half, XRt_half, WH_half, IoW_half, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
                   M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):

    if dim=="3D":
        WF_psi_s = ( (H0*delta_h_sv*Ww_half*psi_1_half - H0*delta_h_sv*Ww*psi_1_n0)/(0.5*dt)\
                    +( delta_h_sv*( 0.5*VoW_half*( (psi_1_half.dx(0)**2)*M11 + dot(hat_psi_star.dx(0),(2.0*MN1*psi_1_half.dx(0)+dot(MNN,hat_psi_star.dx(0)))) )\
                                  + 0.5* Ww_half*( (psi_1_half.dx(1)**2)*M11 + dot(hat_psi_star.dx(1),(2.0*MN1*psi_1_half.dx(1)+dot(MNN,hat_psi_star.dx(1)))) )\
                                       + Uu_half*( psi_1_half.dx(0) * (M11*psi_1_half.dx(1) + dot(MN1,hat_psi_star.dx(1)))\
                                                   + dot(hat_psi_star.dx(0), (MN1*psi_1_half.dx(1) + dot(MNN,hat_psi_star.dx(1))))) )\
                      -(VoW_half*delta_h_sv.dx(0) + Uu_half*delta_h_sv.dx(1)) * ( psi_1_half.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                                                 + dot(hat_psi_star.dx(0), (DN1*psi_1_half + dot(DNN, hat_psi_star))) )\
                      -( Ww_half*delta_h_sv.dx(1) + Uu_half*delta_h_sv.dx(0)) * ( psi_1_half.dx(1)*(D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                                                 + dot(hat_psi_star.dx(1), (DN1*psi_1_half + dot(DNN, hat_psi_star))) )\
                      +(1.0/h_n0) * ( psi_1_half*psi_1_half*S11+ 2.0*dot(hat_psi_star,SN1)*psi_1_half + dot(hat_psi_star,dot(SNN,hat_psi_star)) ) * \
                                    ( delta_h_sv.dx(0)*(VoW_half*h_n0.dx(0)+Uu_half*h_n0.dx(1)) + delta_h_sv.dx(1)*(Ww_half*h_n0.dx(1)+Uu_half*h_n0.dx(0))\
                                      -(delta_h_sv/(2.0*h_n0))*( VoW_half*h_n0.dx(0)**2 + Ww_half*h_n0.dx(1)**2 + 2.0*Uu_half*h_n0.dx(0)*h_n0.dx(1)) )\
-(0.5*delta_h_sv*H0*H0/(h_n0**2)) * ( psi_1_half*psi_1_half*A11+ 2.0*dot(hat_psi_star,AN1)*psi_1_half + dot(hat_psi_star,dot(ANN,hat_psi_star)) ) * \
                                    ( FWF*(VoW_half*b.dx(0)**2 + Ww_half*b.dx(1)**2 + 2.0*Uu_half*b.dx(0)*b.dx(1)) + Ww_half)\
                      + (H0/h_n0) * ( psi_1_half*psi_1_half*C11+ 2.0*dot(hat_psi_star,CN1)*psi_1_half + dot(hat_psi_star,dot(CNN,hat_psi_star))) *\
                                    ( delta_h_sv.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + delta_h_sv.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0))\
                                     -(delta_h_sv/h_n0)*(h_n0.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + h_n0.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0))) )\
                      +H0*g*Ww_half*delta_h_sv*h_n0 -H0*g*WH_half*delta_h_sv - H0*psi_1_half*XRt_half*delta_h_sv.dx(0)) )*dx\
                    + (Lw*dWM_half_dt*delta_h_sv*(psi_1_half*I1 + dot(hat_psi_star,IN)))*ds(1)

        # checked by yl. JCP (D.1), full weak form!
        # checked again against SE step2
        # Implement MSA: FWF=0 and Bij=Cij=0.

        WF_hat_psi_star= (h_n0 * VoW_half * elem_mult(delta_hat_psi_sv.dx(0),(MN1*psi_1_half.dx(0)+dot(MNN,hat_psi_star.dx(0))))\
                         + h_n0 * Ww_half * elem_mult(delta_hat_psi_sv.dx(1),(MN1*psi_1_half.dx(1)+dot(MNN,hat_psi_star.dx(1))))\
                         + h_n0 * Uu_half * ( elem_mult((dot(hat_psi_star.dx(0),MNN)+psi_1_half.dx(0)*M1N),delta_hat_psi_sv.dx(1))\
                                                + elem_mult(delta_hat_psi_sv.dx(0),(MN1*psi_1_half.dx(1)+dot(MNN,hat_psi_star.dx(1)))) )\
                        -(VoW_half*h_n0.dx(0) + Uu_half*h_n0.dx(1)) * (elem_mult(delta_hat_psi_sv,(psi_1_half.dx(0)*D1N+dot(DNN.T,hat_psi_star.dx(0))))\
                                                                     + elem_mult(delta_hat_psi_sv.dx(0),(DN1*psi_1_half+dot(DNN,hat_psi_star))))\
                        -( Ww_half*h_n0.dx(1) + Uu_half*h_n0.dx(0)) * (elem_mult(delta_hat_psi_sv,(D1N*psi_1_half.dx(1)+dot(DNN.T,hat_psi_star.dx(1))))\
                                                                     + elem_mult(delta_hat_psi_sv.dx(1),(DN1*psi_1_half+dot(DNN,hat_psi_star))))\
                        - H0* (VoW_half*b.dx(0) + Uu_half*b.dx(1)) * (elem_mult(delta_hat_psi_sv,(psi_1_half.dx(0)*B1N+dot(BNN.T,hat_psi_star.dx(0))))\
                                                                     + elem_mult(delta_hat_psi_sv.dx(0),(BN1*psi_1_half+dot(BNN,hat_psi_star))))\
                        - H0* ( Ww_half*b.dx(1) + Uu_half*b.dx(0)) * (elem_mult(delta_hat_psi_sv,(B1N*psi_1_half.dx(1)+dot(BNN.T,hat_psi_star.dx(1))))\
                                                                     + elem_mult(delta_hat_psi_sv.dx(1),(BN1*psi_1_half+dot(BNN,hat_psi_star))))\
                        + (1.0/h_n0) * elem_mult(delta_hat_psi_sv,(SN1*psi_1_half + dot(SNN,hat_psi_star))) * \
                                       ( VoW_half*h_n0.dx(0)**2 + Ww_half*h_n0.dx(1)**2 + 2.0*Uu_half*h_n0.dx(0)*h_n0.dx(1) )\
                      + (H0*H0/h_n0) * elem_mult(delta_hat_psi_sv,(AN1*psi_1_half + dot(ANN,hat_psi_star))) * \
                                       ( FWF*(VoW_half*b.dx(0)**2 + Ww_half*b.dx(1)**2 + 2.0*Uu_half*b.dx(0)*b.dx(1)) + Ww_half)\
                     + (2.0*H0/h_n0) * elem_mult(delta_hat_psi_sv,(CN1*psi_1_half + dot(CNN,hat_psi_star))) * \
                                       ( h_n0.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + h_n0.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0)) ))
        
        # checked by yl. JCP (D.2), full vweak form!
        # checked again against SE step1
        # Implement MSA: FWF=0 and Bij=Cij=0.

    if dim=="2D":
        WF_psi_s = ( (H0*delta_h_sv*Ww_half*psi_1_half - H0*delta_h_sv*Ww*psi_1_n0)/(0.5*dt)\
                    + (delta_h_sv* ((Lw*Lw)*0.5*IoW_half)*( (psi_1_half.dx(0)**2)*M11\
                                                   + dot(hat_psi_star.dx(0),(2.0*MN1*psi_1_half.dx(0)+dot(MNN,hat_psi_star.dx(0)))) )\
                      -(IoW_half*(Lw*Lw)*delta_h_sv.dx(0))*( psi_1_half.dx(0)* (D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                               + dot(hat_psi_star.dx(0), (DN1*psi_1_half + dot(DNN, hat_psi_star))) )\
                      +(1.0/h_n0)*( delta_h_sv.dx(0)*(IoW_half*h_n0.dx(0)*(Lw*Lw))\
                                   -(delta_h_sv/h_n0)*((Lw*Lw)*(h_n0.dx(0)**2)*0.5*IoW_half) ) * (psi_1_half*psi_1_half*S11\
                                                               + 2.0*dot(hat_psi_star,SN1)*psi_1_half + dot(hat_psi_star,dot(SNN,hat_psi_star)))\
                      -(0.5*delta_h_sv*Ww_half*H0*H0/(h_n0**2)) * (psi_1_half*psi_1_half*A11\
                                                               + 2.0*dot(hat_psi_star,AN1)*psi_1_half + dot(hat_psi_star,dot(ANN,hat_psi_star)))\
                      +H0*g*Ww_half*delta_h_sv*h_n0 -H0*g*WH_half*delta_h_sv - H0*psi_1_half*XRt_half*delta_h_sv.dx(0)))*dx\
                    + (Lw*dWM_half_dt*delta_h_sv*(psi_1_half*I1 + dot(hat_psi_star,IN)))*ds(1)                      
        
        # checked by yl. Remove terms containing y compared with 3D. Vv = Lw*Lw. MSA
        # checked again against SE step2 

        WF_hat_psi_star= ((h_n0*IoW_half)*(Lw*Lw)*elem_mult(delta_hat_psi_sv.dx(0),(MN1*psi_1_half.dx(0)+dot(MNN,hat_psi_star.dx(0))))\
                        -((Lw*Lw)*h_n0.dx(0)*IoW_half)*( elem_mult(delta_hat_psi_sv,(psi_1_half.dx(0)*D1N+dot(DNN.T,hat_psi_star.dx(0))))\
                                                       + elem_mult(delta_hat_psi_sv.dx(0),(DN1*psi_1_half+dot(DNN,hat_psi_star)))) \
                        +(1.0/h_n0)*((Lw*Lw)*(h_n0.dx(0)**2)*IoW_half)*elem_mult(delta_hat_psi_sv,(SN1*psi_1_half + dot(SNN,hat_psi_star)))\
                                                + (Ww_half*H0*H0/h_n0)*elem_mult(delta_hat_psi_sv,(AN1*psi_1_half + dot(ANN,hat_psi_star))) )
        
        # checked by yl. Remove terms containing y compared with 3D. Vv = Lw*Lw. MSA
        # checked again against SE step1

    WF_hat_BC_star = (Lw*dWM_half_dt*h_n0*elem_mult(delta_hat_psi_sv,IN))
    
    WF_psi_star = WF_psi_s + sum((WF_hat_psi_star[ind])*dx for ind in range(0,n_z)) + sum((WF_hat_BC_star[ind])*ds(1) for ind in range(0,n_z))

    return WF_psi_star

#--------------------------------------------------------------------------------------------------------------------------#
#                              Step 2 : Update h^{n+1} and psi_i at time t^** simulataneously:                             #
#__________________________________________________________________________________________________________________________#

# dim, n_z, g, H0, Lw, dt, x_coord: constant
# WM: at n time level, known
# WM_half, dWM_half_dy, dWM_half_dt: known
# delta_psi, delta_hat_star: test functions
# h_n0: at n time level, known
# psi_1_half, hat_psi_star: obtained from step1 [= split(w_half)], known
# >>> h_n1, hat_psi_aux = split(w_n1): unknown

# delete: x_coord, WM, WM_half, dWM_half_dy, (dWM_half_dt)
# add: b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF

def WF_h_SV(dim, n_z, Lw, H0, g, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_half, hat_psi_star, hat_psi_aux, 
            dWM_half_dt, Ww_half, Uu_half, VoW_half, XRt_half, IoW_half, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
            M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):
    
    if dim == "3D":
        '''
        WF_h = (H0*delta_psi*(h_n1-h_n0)*Ww_half/dt\
               -0.5*( VoW_half * (h_n0*(psi_1_half.dx(0)*M11 + dot(hat_psi_star.dx(0),MN1))\
                                   +h_n1*(psi_1_half.dx(0)*M11 + dot(hat_psi_aux.dx(0),MN1)))*delta_psi.dx(0)\
                        + Ww_half * (h_n0*(psi_1_half.dx(1)*M11 + dot(hat_psi_star.dx(1),MN1))\
                                   +h_n1*(psi_1_half.dx(1)*M11 + dot(hat_psi_aux.dx(1),MN1)))*delta_psi.dx(1)\
                        + Uu_half * (h_n0*(delta_psi.dx(0)*(M11*psi_1_half.dx(1) + dot(M1N,hat_psi_star.dx(1)))\
                                         +delta_psi.dx(1)*(M11*psi_1_half.dx(0) + dot(MN1, hat_psi_star.dx(0))))\
                                   +h_n1*(delta_psi.dx(0)*(M11*psi_1_half.dx(1) + dot(M1N,hat_psi_aux.dx(1)))\
                                         +delta_psi.dx(1)*(M11*psi_1_half.dx(0) + dot(MN1, hat_psi_aux.dx(0)))))\
                      -( VoW_half*h_n0.dx(0) + Uu_half*h_n0.dx(1) )*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                                   +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_star.dx(0),DN1)))\
                      -( VoW_half*h_n1.dx(0) + Uu_half*h_n1.dx(1) )*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                                   +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_aux.dx(0),DN1)))\
                      -(  Ww_half * h_n0.dx(1) + Uu_half*h_n0.dx(0)   )*( delta_psi.dx(1)*(D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                                   +delta_psi*(psi_1_half.dx(1)*D11 + dot(hat_psi_star.dx(1),DN1)))\
                      -(  Ww_half * h_n1.dx(1) + Uu_half*h_n1.dx(0)   )*( delta_psi.dx(1)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                                   +delta_psi*(psi_1_half.dx(1)*D11 + dot(hat_psi_aux.dx(1),DN1)))\
                      +(1/h_n0)*( (VoW_half*(h_n0.dx(0)**2) + Ww_half*(h_n0.dx(1)**2)\
                                                         + 2.0*Uu_half*h_n0.dx(0)*h_n0.dx(1))*(psi_1_half*S11 + dot(hat_psi_star,SN1))*delta_psi)\
                      +(1/h_n1)*( (VoW_half*(h_n1.dx(0)**2) + Ww_half*(h_n1.dx(1)**2)\
                                                         + 2.0*Uu_half*h_n1.dx(0)*h_n1.dx(1))*(psi_1_half*S11 + dot(hat_psi_aux,SN1))*delta_psi)\
                      +(Ww_half*H0*H0/h_n0)*(psi_1_half*A11 + dot(hat_psi_star,AN1))*delta_psi\
                      +(Ww_half*H0*H0/h_n1)*(psi_1_half*A11 + dot(hat_psi_aux,AN1))*delta_psi\
                      -delta_psi*H0*XRt_half*(h_n0.dx(0)+h_n1.dx(0)) ))*dx\
               -0.5*(delta_psi*Lw*dWM_half_dt*(h_n0*I1 + h_n1*I1))*ds(1)
        
        # checked by yl. JCP (D.3), remove terms containing b
        # missing back slash at line 2
        '''

        WF_h =( (H0*delta_psi*(h_n1-h_n0)*Ww_half)/(0.5*dt) \
                -( h_n0*VoW_half*(psi_1_half.dx(0)*M11 + dot(hat_psi_star.dx(0),MN1))*delta_psi.dx(0)\
                   +Ww_half*h_n0*(psi_1_half.dx(1)*M11 + dot(hat_psi_star.dx(1),MN1))*delta_psi.dx(1) \
                   +Uu_half*h_n0*(delta_psi.dx(0)*(M11*psi_1_half.dx(1) + dot(M1N,hat_psi_star.dx(1)))\
                                             + delta_psi.dx(1)*(M11*psi_1_half.dx(0) + dot(MN1, hat_psi_star.dx(0))))\
                  -( VoW_half*h_n0.dx(0) + Uu_half*h_n0.dx(1) )*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_star)) \
                                                +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_star.dx(0),DN1)))\
                  -(  Ww_half*h_n0.dx(1) + Uu_half*h_n0.dx(0) )*( delta_psi.dx(1)*(D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                +delta_psi*(psi_1_half.dx(1)*D11 + dot(hat_psi_star.dx(1),DN1)))\
                  - H0* ( VoW_half*b.dx(0) + Uu_half*b.dx(1) )*( delta_psi.dx(0)*(B11*psi_1_half + dot(B1N,hat_psi_star)) \
                                                +delta_psi*(psi_1_half.dx(0)*B11 + dot(hat_psi_star.dx(0),BN1)))\
                  - H0* (  Ww_half*b.dx(1) + Uu_half*b.dx(0) )*( delta_psi.dx(1)*(B11*psi_1_half + dot(B1N,hat_psi_star))\
                                                +delta_psi*(psi_1_half.dx(1)*B11 + dot(hat_psi_star.dx(1),BN1)))\
                  +(delta_psi/h_n0) * ( psi_1_half*S11 + dot(hat_psi_star,SN1) ) * \
                                      ( VoW_half*h_n0.dx(0)**2 + Ww_half*h_n0.dx(1)**2 + 2.0*Uu_half*h_n0.dx(0)*h_n0.dx(1) )\
            +(delta_psi*H0*H0/h_n0) * ( psi_1_half*A11 + dot(hat_psi_star,AN1) ) * \
                                      ( FWF*(VoW_half*b.dx(0)**2 + Ww_half*b.dx(1)**2 + 2.0*Uu_half*b.dx(0)*b.dx(1)) + Ww_half )\
           +(2.0*H0*delta_psi/h_n0) * ( psi_1_half*C11 + dot(hat_psi_star,CN1) ) * \
                                      ( h_n0.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + h_n0.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0)) )\
                  - delta_psi*H0*XRt_half*h_n0.dx(0) )\
                -( h_n1*VoW_half*(psi_1_half.dx(0)*M11 + dot(hat_psi_aux.dx(0),MN1))*delta_psi.dx(0)\
                  +Ww_half*h_n1*(psi_1_half.dx(1)*M11 + dot(hat_psi_aux.dx(1),MN1))*delta_psi.dx(1) \
                  +Uu_half*h_n1*(delta_psi.dx(0)*(M11*psi_1_half.dx(1) + dot(M1N,hat_psi_aux.dx(1)))\
                                             + delta_psi.dx(1)*(M11*psi_1_half.dx(0) + dot(MN1, hat_psi_aux.dx(0))))\
                  -( VoW_half*h_n1.dx(0) + Uu_half*h_n1.dx(1) )*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_aux)) \
                                                +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_aux.dx(0),DN1)))\
                  -(  Ww_half*h_n1.dx(1) + Uu_half*h_n1.dx(0) )*( delta_psi.dx(1)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                +delta_psi*(psi_1_half.dx(1)*D11 + dot(hat_psi_aux.dx(1),DN1)))\
                  - H0* ( VoW_half*b.dx(0) + Uu_half*b.dx(1) )*( delta_psi.dx(0)*(B11*psi_1_half + dot(B1N,hat_psi_aux)) \
                                                +delta_psi*(psi_1_half.dx(0)*B11 + dot(hat_psi_aux.dx(0),BN1)))\
                  - H0* (  Ww_half*b.dx(1) + Uu_half*b.dx(0) )*( delta_psi.dx(1)*(B11*psi_1_half + dot(B1N,hat_psi_aux))\
                                                +delta_psi*(psi_1_half.dx(1)*B11 + dot(hat_psi_aux.dx(1),BN1)))\
                  +(delta_psi/h_n1) * ( psi_1_half*S11 + dot(hat_psi_aux,SN1) ) * \
                                      ( VoW_half*h_n1.dx(0)**2 + Ww_half*h_n1.dx(1)**2 + 2.0*Uu_half*h_n1.dx(0)*h_n1.dx(1) )\
            +(delta_psi*H0*H0/h_n1) * ( psi_1_half*A11 + dot(hat_psi_aux,AN1) ) * \
                                      ( FWF*(VoW_half*b.dx(0)**2 + Ww_half*b.dx(1)**2 + 2.0*Uu_half*b.dx(0)*b.dx(1)) + Ww_half )\
           +(2.0*H0*delta_psi/h_n1) * ( psi_1_half*C11 + dot(hat_psi_aux,CN1) ) * \
                                      ( h_n1.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + h_n1.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0)) )\
                  - delta_psi*H0*XRt_half*h_n1.dx(0) ) )*dx \
                - (delta_psi*Lw*dWM_half_dt*(h_n0+h_n1)*I1)*ds(1)

        # yl reformulate using SE step1, full weak form!
        # Implement MSA: FWF=0 and Bij=Cij=0.

        WF_hat_psi_aux= (h_n1*VoW_half * elem_mult(delta_hat_star.dx(0),(MN1*psi_1_half.dx(0)+dot(MNN,hat_psi_aux.dx(0))))\
                            + Ww_half *h_n1 * elem_mult(delta_hat_star.dx(1),(MN1*psi_1_half.dx(1)+dot(MNN,hat_psi_aux.dx(1))))\
                            + Uu_half *h_n1 * ( elem_mult((dot(hat_psi_aux.dx(0),MNN)+psi_1_half.dx(0)*M1N),delta_hat_star.dx(1))\
                                          + elem_mult(delta_hat_star.dx(0),(MN1*psi_1_half.dx(1)+dot(MNN,hat_psi_aux.dx(1)))))\
                        -(VoW_half*h_n1.dx(0) + Uu_half*h_n1.dx(1)) * (elem_mult(delta_hat_star,(psi_1_half.dx(0)*D1N+dot(DNN.T,hat_psi_aux.dx(0))))\
                                                                     + elem_mult(delta_hat_star.dx(0),(DN1*psi_1_half+dot(DNN,hat_psi_aux))))\
                        -( Ww_half*h_n1.dx(1) + Uu_half*h_n1.dx(0)) * (elem_mult(delta_hat_star,(D1N*psi_1_half.dx(1)+dot(DNN.T,hat_psi_aux.dx(1))))\
                                                                     + elem_mult(delta_hat_star.dx(1),(DN1*psi_1_half+dot(DNN,hat_psi_aux))))\
                        - H0* (VoW_half*b.dx(0) + Uu_half*b.dx(1)) * (elem_mult(delta_hat_star,(psi_1_half.dx(0)*B1N+dot(BNN.T,hat_psi_aux.dx(0))))\
                                                                     + elem_mult(delta_hat_star.dx(0),(BN1*psi_1_half+dot(BNN,hat_psi_aux))))\
                        - H0* ( Ww_half*b.dx(1) + Uu_half*b.dx(0)) * (elem_mult(delta_hat_star,(B1N*psi_1_half.dx(1)+dot(BNN.T,hat_psi_aux.dx(1))))\
                                                                     + elem_mult(delta_hat_star.dx(1),(BN1*psi_1_half+dot(BNN,hat_psi_aux))))\
                        +(1.0/h_n1) * elem_mult(delta_hat_star,(SN1*psi_1_half + dot(SNN,hat_psi_aux))) * \
                                       ( VoW_half*h_n1.dx(0)**2 + Ww_half*h_n1.dx(1)**2 + 2.0*Uu_half*h_n1.dx(0)*h_n1.dx(1) )\
                     + (H0*H0/h_n1) * elem_mult(delta_hat_star,(AN1*psi_1_half + dot(ANN,hat_psi_aux))) * \
                                       ( FWF*(VoW_half*b.dx(0)**2 + Ww_half*b.dx(1)**2 + 2.0*Uu_half*b.dx(0)*b.dx(1)) + Ww_half)\
                 + (2.0*H0*H0/h_n1) * elem_mult(delta_hat_star,(CN1*psi_1_half + dot(CNN,hat_psi_aux))) * \
                                       ( h_n1.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + h_n1.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0)) ))

        # checked by yl. JCP (D.4), full weak form!
        # checked again against SE step1
        # Implement MSA: FWF=0 and Bij=Cij=0.

    elif dim=="2D":
        '''
        WF_h = (H0*delta_psi*(h_n1-h_n0)*Ww_half/dt\
                -0.5*((h_n0*IoW_half)*(Lw*Lw)*(psi_1_half.dx(0)*M11 + dot(hat_psi_star.dx(0),MN1))*delta_psi.dx(0)\
                      -( IoW_half*(Lw*Lw)*h_n0.dx(0))*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                               +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_star.dx(0),DN1)))\
                      +(1/h_n0)*( IoW_half*(Lw*Lw)*(h_n0.dx(0)**2))*(psi_1_half*S11 + dot(hat_psi_star,SN1))*delta_psi\
                      +(Ww_half*H0*H0/h_n0)*(psi_1_half*A11 + dot(hat_psi_star,AN1))*delta_psi\
                      - delta_psi*H0*XRt_half*h_n0.dx(0))\
                -0.5*((h_n1*IoW_half)*(Lw*Lw)*(psi_1_half.dx(0)*M11 + dot(hat_psi_aux.dx(0),MN1))*delta_psi.dx(0)\
                      -( IoW_half*(Lw*Lw)*h_n1.dx(0))*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                               +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_aux.dx(0),DN1)))\
                      +(1/h_n1)*( IoW_half*(Lw*Lw)*(h_n1.dx(0)**2))*(psi_1_half*S11 + dot(hat_psi_aux,SN1))*delta_psi\
                      +(Ww_half*H0*H0/h_n1)*(psi_1_half*A11 + dot(hat_psi_aux,AN1))*delta_psi\
                      - delta_psi*H0*XRt_half*h_n1.dx(0)) )*dx\
                -0.5*( delta_psi*Lw*dWM_half_dt*(h_n0+h_n1)*I1 )*ds(1)
        
        # checked by yl.
        # missing back slash at line 7
        '''

        WF_h =( (H0*delta_psi*(h_n1-h_n0)*Ww_half)/(0.5*dt)\
                -((h_n0*IoW_half)*(Lw*Lw)*(psi_1_half.dx(0)*M11 + dot(hat_psi_star.dx(0),MN1))*delta_psi.dx(0)\
                      -( IoW_half*(Lw*Lw)*h_n0.dx(0))*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_star))\
                                                               +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_star.dx(0),DN1)))\
                      +(1/h_n0)*( IoW_half*(Lw*Lw)*(h_n0.dx(0)**2))*(psi_1_half*S11 + dot(hat_psi_star,SN1))*delta_psi\
                      +(Ww_half*H0*H0/h_n0)*(psi_1_half*A11 + dot(hat_psi_star,AN1))*delta_psi\
                      - delta_psi*H0*XRt_half*h_n0.dx(0))\
                -((h_n1*IoW_half)*(Lw*Lw)*(psi_1_half.dx(0)*M11 + dot(hat_psi_aux.dx(0),MN1))*delta_psi.dx(0)\
                      -( IoW_half*(Lw*Lw)*h_n1.dx(0))*( delta_psi.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                               +delta_psi*(psi_1_half.dx(0)*D11 + dot(hat_psi_aux.dx(0),DN1)))\
                      +(1/h_n1)*( IoW_half*(Lw*Lw)*(h_n1.dx(0)**2))*(psi_1_half*S11 + dot(hat_psi_aux,SN1))*delta_psi\
                      +(Ww_half*H0*H0/h_n1)*(psi_1_half*A11 + dot(hat_psi_aux,AN1))*delta_psi\
                      - delta_psi*H0*XRt_half*h_n1.dx(0)) )*dx\
                -( delta_psi*Lw*dWM_half_dt*(h_n0+h_n1)*I1 )*ds(1)

        # yl reformulate using SE step1. MSA

        WF_hat_psi_aux= ((h_n1*IoW_half)*(Lw*Lw)*elem_mult(delta_hat_star.dx(0),(MN1*psi_1_half.dx(0)+dot(MNN,hat_psi_aux.dx(0))))\
                         -((Lw*Lw)*h_n1.dx(0)*IoW_half)*( elem_mult(delta_hat_star,(psi_1_half.dx(0)*D1N+dot(DNN.T,hat_psi_aux.dx(0))))\
                                                        + elem_mult(delta_hat_star.dx(0),(DN1*psi_1_half+dot(DNN,hat_psi_aux))))\
                         +(1.0/h_n1)*((Lw*Lw)*(h_n1.dx(0)**2)*IoW_half)*elem_mult(delta_hat_star,(SN1*psi_1_half+ dot(SNN,hat_psi_aux)))\
                         + (Ww_half*H0*H0/h_n1)*elem_mult(delta_hat_star,(AN1*psi_1_half+dot(ANN,hat_psi_aux))))

        # checked by yl.
        # checked again against SE step1. MSA

    WF_hat_BC_aux = (Lw*dWM_half_dt*h_n1*elem_mult(delta_hat_star,IN))

    WF_h_psi = WF_h + sum((WF_hat_psi_aux[ind])*dx for ind in range(0,n_z)) + sum((WF_hat_BC_aux[ind])*ds(1) for ind in range(0,n_z))

    return WF_h_psi

#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 3 : Update psi_1 at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# dim, g, H, H0, Lw, dt, x_coord: constant
# WM_half, dWM_half_dy, dWM_half_dt: at n+1/2 time level
# WM_n1: at n+1 time level
# delta_h: test function
# psi_1_half: obtained from step1, known
# h_n1, hat_psi_aux: obtained from step2, known
# >>> psi_1: trial function

# delete: H, x_coord, WM_n1, WM_half, (dWM_half_dt), dWM_half_dy
# add: b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF

def WF_psi_n1_SV(dim, H0, g, delta_h, Lw, dt, psi_1_half, psi_1, dWM_half_dt, hat_psi_aux, h_n1, 
                 Ww_n1, Ww_half, Uu_half, VoW_half, XRt_half, WH_half, IoW_half, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
                 M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):
    
    if dim=="3D":
        a_psi_1 = (H0*delta_h*Ww_n1*psi_1/(0.5*dt))*dx 

        # modified. checked by yl. LHS of (D.5)

        L_psi_1 = -(-H0*delta_h*Ww_half*psi_1_half/(0.5*dt)\
                    +(delta_h*( 0.5*VoW_half*((psi_1_half.dx(0)**2)*M11 + dot(hat_psi_aux.dx(0), (2.0*MN1*psi_1_half.dx(0) + dot(MNN,hat_psi_aux.dx(0)))))\
                               + 0.5*Ww_half*((psi_1_half.dx(1)**2)*M11 + dot(hat_psi_aux.dx(1), (2.0*MN1*psi_1_half.dx(1) + dot(MNN,hat_psi_aux.dx(1)))))\
                                   +Uu_half *( psi_1_half.dx(0)*(M11*psi_1_half.dx(1) + dot(MN1,hat_psi_aux.dx(1))) \
                                                + dot(hat_psi_aux.dx(0), (MN1*psi_1_half.dx(1) + dot(MNN,hat_psi_aux.dx(1))))) )\
                            -(VoW_half*delta_h.dx(0) + Uu_half*delta_h.dx(1)) * ( psi_1_half.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                                        + dot(hat_psi_aux.dx(0), (DN1*psi_1_half + dot(DNN, hat_psi_aux))) )\
                            -( Ww_half*delta_h.dx(1) + Uu_half*delta_h.dx(0)) * ( psi_1_half.dx(1)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                                        + dot(hat_psi_aux.dx(1), (DN1*psi_1_half + dot(DNN, hat_psi_aux))) )\
                            +(1.0/h_n1) * ( psi_1_half*psi_1_half*S11 + 2.0*dot(hat_psi_aux,SN1)*psi_1_half + dot(hat_psi_aux,dot(SNN,hat_psi_aux)) )*\
                                          ( delta_h.dx(0)*(VoW_half*h_n1.dx(0)+Uu_half*h_n1.dx(1)) + delta_h.dx(1)*(Ww_half*h_n1.dx(1)+Uu_half*h_n1.dx(0))\
                                            -(delta_h/(2.0*h_n1))*(VoW_half*h_n1.dx(0)**2 + Ww_half*h_n1.dx(1)**2 + 2.0*Uu_half*h_n1.dx(0)*h_n1.dx(1)) )\
         -(0.5*delta_h*H0*H0/(h_n1**2)) * ( psi_1_half*psi_1_half*A11 + 2.0*dot(hat_psi_aux,AN1)*psi_1_half + dot(hat_psi_aux,dot(ANN,hat_psi_aux)) )*\
                                          ( FWF*(VoW_half*b.dx(0)**2 + Ww_half*b.dx(1)**2 + 2.0*Uu_half*b.dx(0)*b.dx(1)) + Ww_half)\
                             +(H0/h_n1) * ( psi_1_half*psi_1_half*C11 + 2.0*dot(hat_psi_aux,CN1)*psi_1_half + dot(hat_psi_aux,dot(CNN,hat_psi_aux)) )*\
                                          ( delta_h.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + delta_h.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0))\
                                            -(delta_h/h_n1)*(h_n1.dx(0)*(VoW_half*b.dx(0)+Uu_half*b.dx(1)) + h_n1.dx(1)*(Ww_half*b.dx(1)+Uu_half*b.dx(0))) )\
                            +H0*g*Ww_half*delta_h*h_n1 -H0*g*WH_half*delta_h - H0*psi_1_half*XRt_half*delta_h.dx(0)) )*dx\
                    -(Lw*dWM_half_dt*delta_h*(psi_1_half*I1 + dot(hat_psi_aux,IN)))*ds(1)

        # checked by yl. RHS of (D.5), full weak form!
        # checked again against SE step2
        # Implement MSA: FWF=0 and Bij=Cij=0.


    elif dim=="2D":
        a_psi_1 = (H0*delta_h*Ww_n1*psi_1/(0.5*dt))*dx
        
        L_psi_1 = -(-H0*delta_h*Ww_half*psi_1_half/(0.5*dt) \
                    +(delta_h*((Lw*Lw)*0.5*IoW_half)*((psi_1_half.dx(0)**2)*M11 +dot(hat_psi_aux.dx(0),(2.0*MN1*psi_1_half.dx(0) + dot(MNN,hat_psi_aux.dx(0)))))\
                            -(IoW_half*(Lw*Lw)*delta_h.dx(0))*( psi_1_half.dx(0)*(D11*psi_1_half + dot(D1N,hat_psi_aux))\
                                                              +dot(hat_psi_aux.dx(0), (DN1*psi_1_half + dot(DNN, hat_psi_aux))) )\
                            +(1.0/h_n1)*(delta_h.dx(0)*(IoW_half*h_n1.dx(0)*(Lw*Lw))\
                                   -(delta_h/h_n1)*((Lw*Lw)*(h_n1.dx(0)**2)*0.5*IoW_half))*(psi_1_half*psi_1_half*S11\
                                                                  +2.0*dot(hat_psi_aux,SN1)*psi_1_half+dot(hat_psi_aux,dot(SNN,hat_psi_aux)) )\
                            -(0.5*delta_h*Ww_half*H0*H0/(h_n1**2))*(psi_1_half*psi_1_half*A11\
                                                                  +2.0*dot(hat_psi_aux,AN1)*psi_1_half+dot(hat_psi_aux,dot(ANN,hat_psi_aux)) )\
                            +H0*g*Ww_half*delta_h*h_n1 -H0*g*WH_half*delta_h - H0*psi_1_half*XRt_half*delta_h.dx(0)) )*dx\
                    -(Lw*dWM_half_dt*delta_h*(psi_1_half*I1 + dot(hat_psi_aux,IN)))*ds(1)

        # modified. checked by yl. MSA
        # checked again against SE step2

    return a_psi_1, L_psi_1


#----------------------------------------------------------------------------------------------------------------------#
#                                        Step 4 : Update psi_i at time t^{n+1}:                                        #
#______________________________________________________________________________________________________________________#

# dim, H, H0, n_z, Lw, dt, x_coord: constant
# WM, dWM_dy, dWM_dt: at n time level
# delta_hat_psi: test function
# h_n0, psi_1_n0: at n time level, known
# >>> hat_psi: trial function
# the solution is placed in hat_psi_n0, just for the sake of output.

# delete: H, WM, x_coord, (dWM_dt), dWM_dy

def WF_hat_psi_SV(dim, n_z, Lw, H0, dt, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                  dWM_dt, Uu, Ww, VoW, IoW,
                  M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN):
    
    if dim=="3D":
        a_hat_psi =( h_n0*VoW*elem_mult(delta_hat_psi.dx(0), dot(MNN,hat_psi.dx(0)))\
                     +Ww*h_n0*elem_mult(delta_hat_psi.dx(1),dot(MNN,hat_psi.dx(1)))\
                     +Uu*h_n0*( elem_mult(dot(hat_psi.dx(0),MNN),delta_hat_psi.dx(1)) + elem_mult(delta_hat_psi.dx(0),dot(MNN,hat_psi.dx(1))))\
                     -( VoW*h_n0.dx(0) + Uu*h_n0.dx(1))*( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(0)))\
                                                             + elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_psi)) )\
                     -(   Ww*h_n0.dx(1) + Uu*h_n0.dx(0)  ) *( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(1)))\
                                                             + elem_mult(delta_hat_psi.dx(1),dot(DNN,hat_psi)) )\
                     +(1.0/h_n0)*( VoW*(h_n0.dx(0)**2) + Ww*(h_n0.dx(1)**2) + 2.0*Uu*h_n0.dx(0)*h_n0.dx(1) )*elem_mult(delta_hat_psi,dot(SNN,hat_psi))\
                     +(Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,dot(ANN,hat_psi)))
        
        # checked by yl. Exactly the same as Step 3 of SE.

        L_hat_psi =-( h_n0*VoW*elem_mult(delta_hat_psi.dx(0), MN1*psi_1_n0.dx(0))\
                         + Ww*h_n0*elem_mult(delta_hat_psi.dx(1), MN1*psi_1_n0.dx(1))\
                         +Uu*h_n0*(elem_mult(delta_hat_psi.dx(1), MN1*psi_1_n0.dx(0))\
                                   + elem_mult(delta_hat_psi.dx(0),MN1*psi_1_n0.dx(1)))\
                      -( VoW*h_n0.dx(0) + Uu*h_n0.dx(1))* ( elem_mult(delta_hat_psi, D1N*psi_1_n0.dx(0))\
                                                              + elem_mult(delta_hat_psi.dx(0),DN1*psi_1_n0) )\
                      -(   Ww*h_n0.dx(1) + Uu*h_n0.dx(0)  ) * ( elem_mult(delta_hat_psi, D1N*psi_1_n0.dx(1))\
                                                              + elem_mult(delta_hat_psi.dx(1),DN1*psi_1_n0))\
                      +(1.0/h_n0)*( VoW*h_n0.dx(0)**2 + Ww*h_n0.dx(1)**2 + 2.0*Uu*h_n0.dx(0)*h_n0.dx(1) )*elem_mult(delta_hat_psi,SN1*psi_1_n0)\
                      +(Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,AN1*psi_1_n0))
        
        # checked by yl. Exactly the same as Step 3 of SE.
           
    elif dim=="2D":
        a_hat_psi =(h_n0*(Lw*Lw*IoW)*elem_mult(delta_hat_psi.dx(0), dot(MNN,hat_psi.dx(0)))\
                     -(Lw*Lw*IoW)*h_n0.dx(0)*( elem_mult(delta_hat_psi, dot(DNN.T,hat_psi.dx(0))) + elem_mult(delta_hat_psi.dx(0),dot(DNN,hat_psi)) )\
                     +(1.0/h_n0)*((Lw*Lw)*(h_n0.dx(0)**2)*IoW)*elem_mult(delta_hat_psi,dot(SNN,hat_psi))\
                     + (Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,dot(ANN,hat_psi)))

        L_hat_psi =-((h_n0*IoW)*(Lw*Lw)*elem_mult(delta_hat_psi.dx(0), MN1*psi_1_n0.dx(0))\
                      -((Lw*Lw)*h_n0.dx(0)*IoW)*( elem_mult(delta_hat_psi, D1N*psi_1_n0.dx(0))\
                                               + elem_mult(delta_hat_psi.dx(0),DN1*psi_1_n0) )\
                      +(1.0/h_n0)*( (Lw*Lw)*(h_n0.dx(0)**2)*IoW)*elem_mult(delta_hat_psi,SN1*psi_1_n0)\
                      + (Ww*H0*H0/h_n0)*elem_mult(delta_hat_psi,AN1*psi_1_n0) )
        
        # checked by yl. Exactly the same as Step 3 of SE.

    hat_psi_BC = -(Lw*dWM_dt*h_n0*elem_mult(delta_hat_psi,IN))
    A_hat = sum((a_hat_psi[ind])*dx for ind in range(0,n_z))
    L_hat = sum((L_hat_psi[ind])*dx for ind in range(0,n_z)) + sum((hat_psi_BC[ind])*ds(1) for ind in range(0,n_z))

    return A_hat, L_hat

