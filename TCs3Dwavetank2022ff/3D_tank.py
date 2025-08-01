# - The updated code for 2D and 3D potential-flow simulations using manually-derived weak forms for SE/SV
# - In the vertical direction, the layers are distributed unevenly according to the GLL points
# - The commented part is for 3D visualisation via ParaView coded by FG.
# - The old 3D TC2 (by FG) has been replaced by 2D TC1.
# - The time variables in wavemaker-related functions are replaced by Firedrake Constant objects. [July 2025]

import pdb
import time
import numpy as np
import os.path

import sympy as sp
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule

from firedrake import *
from firedrake.petsc import PETSc
import solvers_full as YL_solvers
from savings import *

# monitor memory usage 
import psutil, os
def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2  # Resident Set Size in MB
    print(f"{label} Memory usage = {mem_mb:.2f} MB")

# Input the test case you are going to run below:
# TC1/TC2/TC3/TC4/TCU
case = 'TC2'
assert case.upper() in ['TC1', 'TC2', 'TC3', 'TC4', 'TCU'], "Incorrect input!"
PETSc.Sys.Print('Setting up test case %3s...' % case)

start_time = time.perf_counter()

"""
    ****************************************
    *               Settings               *
    **************************************** """
if case in ['TC1', 'TC2']:
    if case == 'TC1':
        from settings_TC1 import *
    else:
        from settings_TC2 import *
    input_data, scheme, dim, save_path, bottom, FWF, save_pvd = test_case()
    k, H0, xb, sb, H_expr, Hend, Lx, Ly, Lw, res_x, res_y, n_z = domain(bottom)
    g, w, Tw, gamma, t_stop, Aa, Bb, WM_expr, dWM_dt_expr, dWM_dy_expr, h_ex_expr, phis_ex_expr, phii_ex_expr = wavemaker(k, dim, H0, Ly, Lw, input_data)
    T0, t, dt, Tend, dt_save = set_time(Lx, res_x, Ly, res_y, H0, n_z, Tw)
else:
    if case=='TC3': 
        from settings_TC3 import *
    elif case=='TC4':
        from settings_TC4 import *
    else:
        from settings_User import *
    input_data, scheme, dim, save_path, bottom, FWF, save_pvd = test_case()
    H0, xb, sb, H_expr, Hend, Lx, Ly, Lw, res_x, res_y, n_z = domain(bottom)
    g, lamb, k, w, Tw, gamma, t_stop, WM_expr, dWM_dt_expr, dWM_dy_expr = wavemaker(dim, H0, Ly, Lw, input_data)
    T0, t, dt, Tend, dt_save = set_time(Tw)

PETSc.Sys.Print('...settings loaded!')

# Create the directory to store output files
try:
    os.makedirs(save_path, exist_ok = True)
    PETSc.Sys.Print("Directory created successfully!")
except OSError as error:
    PETSc.Sys.Print("Directory can not be created!")
    
"""
    ****************************************************
    *               Definition of the mesh             *
    **************************************************** """
PETSc.Sys.Print('Creation of the mesh across %d processes...' % COMM_WORLD.size)

#_________________ Vertical discretization ________________#
Nz = n_z+1         # Number of point in one vertical element

#________________ Horizontal discretization _______________#
Nx = round(Lx/res_x)    # Number of elements in x (round to the nearest integer)
Ny = round(Ly/res_y)    # Number of elements in y

#___________________________ Mesh _________________________#
if dim=="2D":                                   #(x,z)-waves
    # Generate a uniform mesh of an interval (L_x).
    hor_mesh = IntervalMesh(Nx,Lx) 
else:                                         #(x,y,z)-waves
    # Generate a rectangular mesh.
    hor_mesh = RectangleMesh(Nx,Ny,Lx,Ly,quadrilateral=True)
    # quadrilateral – (optional), creates quadrilateral mesh, defaults to False

x = SpatialCoordinate(hor_mesh)

PETSc.Sys.Print('...mesh created!')

PETSc.Sys.Print('Definition of the function...')

"""
    *************************************************
    *       Definition of the function spaces       *
    ************************************************* """
#___________________ For h and psi_1 ___________________#
V = FunctionSpace(hor_mesh, "CG", 1)
#_____________________ For hat_psi _____________________#
Vec = VectorFunctionSpace(hor_mesh, "CG", 1, dim=n_z)
# We might want the number of components in the vector to differ from the geometric dimension of the mesh. 
# We can do this by passing a value for the dim argument to the VectorFunctionSpace() constructor.
#_________________ Mixed function space ________________#
V_mixed = V*Vec # to solve simultaneous weak formulations

"""
    ******************************************************
    *            Definition of the functions             *
    ****************************************************** """

if scheme=="SE": #_________ Symplectic-Euler scheme _________#
    #______________________ At time t^n _____________________#
    h_n0 = Function(V)                                   # h^n
    psi_1_n0 = Function(V)                           # psi_1^n
    hat_psi_n0 = Function(Vec)                     # hat_psi^n

    #________________ At time t^{n+1} and t^* _______________#
    psi_1_n1 = Function(V)                       # psi_1^{n+1}
    w_n1 = Function(V_mixed)
    h_n1, hat_psi_star = split(w_n1)      # h^{n+1}, hat_psi^*
    hat_psi_n1 = Function(Vec)    # to visualise hat_psi^{n+1}
else: #________________ Stormer-Verlet scheme _______________#
    #______________________ At time t^n _____________________#
    h_n0 = Function(V)                                   # h^n
    psi_1_n0 = Function(V)                           # psi_1^n
    hat_psi_n0 = Function(Vec)                     # hat_psi^n

    #_______________ At time t^{n+1/2} and t^* ______________#
    w_half = Function(V_mixed)        # to obtain psi^{n+1/2},
    psi_1_half, hat_psi_star = split(w_half)   # and hat_psi^*

    #_______________ At time t^{n+1} and t^** _______________#
    psi_1_n1 = Function(V)                       # psi_1^{n+1}
    w_n1 = Function(V_mixed)              # to obtain h^{n+1},
    h_n1, hat_psi_aux = split(w_n1)         # and hat_psi^{**}
    hat_psi_n1 = Function(Vec)    # to visualise hat_psi^{n+1}

#________________ Beach topography b(x,y)____________________#
b = Function(V)                                       # b(x,y)

#_______________________ Depth at rest ______________________#
H = Function(V)                                         # H(x)

#_________________________ Wavemaker ________________________#
WM = Function(V)                                  # R(x,y;t^n)
dWM_dt = Function(V)                               # (dR/dt)^n
dWM_dy = Function(V)                               # (dR/dy)^n
if scheme=="SV":                         # For Stormer-Verlet:
    WM_half = Function(V)                   # R(x,y;t^{n+1/2})
    dWM_half_dt = Function(V)                # (dR/dt)^{n+1/2}
    dWM_half_dy = Function(V)                # (dR/dy)^{n+1/2}
WM_n1 = Function(V)                           # R(x,y;t^{n+1})
dWM_n1_dt = Function(V)                        # (dR/dt)^{n+1}
dWM_n1_dy = Function(V)                        # (dR/dy)^{n+1}

#______________________ Trial functions _____________________#
psi_1 = TrialFunction(V)      # psi_1^{n+1} for linear solvers
hat_psi = TrialFunction(Vec)# hat_psi^{n+1} for linear solvers

#_______________________ Test functions _____________________#
delta_h = TestFunction(V)                         # from dH/dh
delta_hat_psi = TestFunction(Vec)           # from dH/dhat_psi
w_t = TestFunction(V_mixed)                # from dH/dpsi_1...
delta_psi, delta_hat_star = split(w_t)    # ...and dH/dhat_psi
# yl update:
if scheme=="SV": 
    w_t_sv = TestFunction(V_mixed)
    delta_h_sv, delta_hat_psi_sv = split(w_t_sv)

if case in ['TC1', 'TC2']:
    phii_z = Function(V) # for initialising standing wave solution

# ------SE------
# step1: use delta_psi and delta_hat_star to solve simultaneously for h^n+1 and psi_hat^*
# step2: use delta_h to solve for psi_1^(n+1)
# step3: use delta_hat_psi to update psi_hat^(n+1) using Laplace Eq.
# ------SV------
# step1: use delta_h_sv and delta_hat_psi_sv to solve simultaneously for psi_1^half and psi_hat^*
# step2: use delta_psi and delta_hat_star to solve simultaneously for h^n+1 and psi_hat^**
# step3: use delta_h to solve for psi_1^(n+1)
# step4: use delta_hat_psi to update psi_hat^(n+1) using Laplace Eq.

# yl update: XUVW
Xx=Function(V) 

Uu=Function(V)
Ww=Function(V)
Vv=Function(V)
VoW=Function(V)
IoW=Function(V)
WH=Function(V)  # W x H
XRt=Function(V) # X x dR/dt

Ww_n1=Function(V) # for SE and SV

if scheme=="SV":  # for SV
    Uu_half =Function(V)
    Ww_half =Function(V)
    Vv_half =Function(V)
    VoW_half=Function(V)
    IoW_half=Function(V)
    WH_half =Function(V)  # W x H
    XRt_half=Function(V)  # X x dR/dt

PETSc.Sys.Print('...functions created!')

PETSc.Sys.Print('Initalisation of the functions...')
"""
    ***********************************************************************************
    *                          Initialisation of the Functions                        *
    ***********************************************************************************"""
#---------------------------- Topography ----------------------------#             
H_expr(H,x)     # Water depth at rest H(x,y)
b.assign(H0-H)  # Bathymetry b(x,y)

#----------------------------------------------------------------------------------------#
#                                       Wavemaker                                        #
#----------------------------------------------------------------------------------------#
if input_data=="measurements":#------------- Interpolate measurements -------------#
    t_wm, wm_data, t_vel, vel_data = load_wavemaker(dt)
    PETSc.Sys.Print('-> Wavemaker data loaded')
    Rt = round(np.interp(T0, t_wm, wm_data), 4)
    Rt_const = Constant(Rt)
    Rt_t = round(np.interp(T0, t_vel, vel_data), 4)
    Rt_t_const = Constant(Rt_t)
    WM_expr(WM, x, Rt_const)
    dWM_dt_expr(dWM_dt, x, Rt_t_const)
    
else:
    t_const = Constant(T0)
    t_half_const = Constant(T0+0.5*dt)
    # yl update
    WM_expr(WM,x,t_const,t_stop)                  # \tilde{R}(x,y;t)
    dWM_dt_expr(dWM_dt,x,t_const,t_stop)          # d\tilde{R}/dt  
    dWM_dy_expr(dWM_dy,x,t_const,t_stop)          # d\tilde{R}/dy

#----------------------------------------------------------------------------------------#
#                               Solutions Initialization                                 #
#----------------------------------------------------------------------------------------#
#____________________________ Initialization of Depth ___________________________________#
if case in ['TC1', 'TC2']:
    h_ex_expr(h_n0,x,t)
    w_n1.sub(0).assign(h_n0)
else:
    h_n0.assign(H)        # h(x,y;t=0) = H(x)                                                  
    w_n1.sub(0).assign(H) # Extract the ith sub Function of this Function. In this case, h^{n+1}.

#_____________________ Velocity pot. at the surface: phi(x,y,z=h;t) _____________________#
if case in ['TC1', 'TC2']:
    phis_ex_expr(psi_1_n0,x,t)

# yl update: XUVW
Xx.interpolate(x[0]-Lw)
Uu.interpolate(Xx*dWM_dy)
Ww.interpolate(Lw-WM)
Vv.interpolate(Lw*Lw+Uu*Uu)
VoW.interpolate(Vv/Ww)
IoW.interpolate(1/Ww)
WH.interpolate(Ww*H)
XRt.interpolate(Xx*dWM_dt)

if scheme=='SV' and input_data=='measurements':
    Uu_half = Constant(0.0)
    Vv_half = Constant(Lw*Lw)

PETSc.Sys.Print('...functions initialised!')

PETSc.Sys.Print('Assembling z-matrices...')

"""
    ************************
    * Compute the matrices *
    ************************ """
#_______ Initialization ______#
A = np.zeros((Nz,Nz))
B = np.zeros((Nz,Nz)) # FWF
C = np.zeros((Nz,Nz)) # FWF
M = np.zeros((Nz,Nz))
D = np.zeros((Nz,Nz))
S = np.zeros((Nz,Nz))
Ik = np.zeros((Nz,1))

# construction of Lagrange polynomials
varphi=[]
z = sp.Symbol('z', positive = True)
#-------------GLL points--------------
fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), Nz)
zeros = fiat_rule.get_points()
z_k_rev = [zeros.item(n_z-k) for k in range(Nz)] 
z_k =[H0*z_k_rev[i] for i in range(len(z_k_rev))]
#-------------------------------------
for i in range(Nz):
    index = list(range(Nz))
    index.pop(i)
    varphi.append(sp.prod([(z-z_k[j])/(z_k[i]-z_k[j]) for j in index]))

#____ Filling the matrices ___#
for i in range(0,Nz):
    for j in range(0,Nz):
        expr_A = sp.diff(varphi[i],z)*sp.diff(varphi[j],z)
        expr_M = varphi[i]*varphi[j]
        expr_D = z*varphi[i]*sp.diff(varphi[j],z)
        expr_S = z*z*sp.diff(varphi[i],z)*sp.diff(varphi[j],z)
        A[i,j] = sp.integrate(expr_A, (z,0,H0))
        M[i,j] = sp.integrate(expr_M, (z,0,H0))
        D[i,j] = sp.integrate(expr_D, (z,0,H0))
        S[i,j] = sp.integrate(expr_S, (z,0,H0))
    Ik[i] = sp.integrate(varphi[i],(z,0,H0))

#________ Submatrices ________#
A11 = A[0,0]
A1N = as_tensor(A[0,1:])
AN1 = as_tensor(A[1:,0])
ANN = as_tensor(A[1:,1:])

M11 = M[0,0]
M1N = as_tensor(M[0,1:])
MN1 = as_tensor(M[1:,0])
MNN = as_tensor(M[1:,1:])

D11 = D[0,0]
D1N = as_tensor(D[0,1:])
DN1 = as_tensor(D[1:,0])
DNN = as_tensor(D[1:,1:])
# D1N[i]!=DN1[i]

S11 = S[0,0]
S1N = as_tensor(S[0,1:])
SN1 = as_tensor(S[1:,0])
SNN = as_tensor(S[1:,1:])

I1 = Ik[0,0]
IN = as_tensor(Ik[1:,0])

# yl added: full weak forms
if FWF==1:
    for i in range(0,Nz):
        for j in range(0,Nz):
            expr_B = varphi[i]*sp.diff(varphi[j],z)
            expr_C = z*sp.diff(varphi[i],z)*sp.diff(varphi[j],z)
            B[i,j] = sp.integrate(expr_B, (z,0,H0))
            C[i,j] = sp.integrate(expr_C, (z,0,H0))

B11 = B[0,0]
B1N = as_tensor(B[0,1:])
BN1 = as_tensor(B[1:,0])
BNN = as_tensor(B[1:,1:])

C11 = C[0,0]
C1N = as_tensor(C[0,1:])
CN1 = as_tensor(C[1:,0])
CNN = as_tensor(C[1:,1:])

PETSc.Sys.Print('... end of assembling!')

PETSc.Sys.Print('Initialisation of the solvers...')

"""
    ************************************************************************************************************************
    *                                                   Weak Formulations                                                  *
    ************************************************************************************************************************ """

if scheme=="SE": #_____________________________________________ Symplectic-Euler ______________________________________________#
    # yl update: XUVW, full 3D WF
    #------------------------ Step 1 : Update h at time t^{n+1} and psi_i at time t^* simulataneously: ------------------------#
    WF_h_psi = YL_solvers.WF_h_SE(dim, n_z, g, H0, Lw, dWM_dt, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_n0, hat_psi_star,
                                  Uu, Ww, VoW, IoW, WH, XRt, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
                                  M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)
    # full 3D WF.
    #----------------------------------------- Step 2 : Update psi_1 at time t^{n+1}: -----------------------------------------#
    A_psi_s, L_psi_s = YL_solvers.WF_psi_SE(dim, g, H0, Lw, dWM_dt, dt, delta_h, psi_1, psi_1_n0, hat_psi_star, h_n1, 
                                            Uu, Ww, Ww_n1, VoW, IoW, WH, XRt, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
                                            M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)

    #----------------------------------------- Step 3 : Update psi_i at time t^{n+1}: -----------------------------------------#
    A_hat, L_hat = YL_solvers.WF_hat_psi_SE(dim, H0, n_z, Lw, dWM_dt, dt, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                                            Uu, Ww, VoW, IoW, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)

elif scheme=="SV":#______________________________________________ Stormer-Verlet ______________________________________________#
    # yl update: XUVW, full 3D WF
    #--------------------------------------- Step 1 : Update psi_1^{n+1/2} and psi_i^*: ---------------------------------------#
    WF_psi_star = YL_solvers.WF_psi_half_SV(dim, n_z, g, H0, Lw, dt, delta_h_sv, delta_hat_psi_sv, psi_1_n0, psi_1_half, hat_psi_star, h_n0, dWM_half_dt, 
                                            Ww, Uu_half, VoW_half, Ww_half, XRt_half, WH_half, IoW_half, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
                                            M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)
    # full 3D WF
    #----------------------------- Step 2 : Update h^{n+1} and psi_i at time t^** simulataneously: ----------------------------#
    WF_h_psi = YL_solvers.WF_h_SV(dim, n_z, Lw, H0, g, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_half, hat_psi_star, hat_psi_aux, 
                                  dWM_half_dt, Ww_half, Uu_half, VoW_half, XRt_half, IoW_half, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
                                  M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)
    # full 3D WF
    #----------------------------------------- Step 3 : Update psi_1 at time t^{n+1}: -----------------------------------------#
    a_psi_1, L_psi_1 = YL_solvers.WF_psi_n1_SV(dim, H0, g, delta_h, Lw, dt, psi_1_half, psi_1, dWM_half_dt, hat_psi_aux, h_n1, 
                                               Ww_n1, Ww_half, Uu_half, VoW_half, XRt_half, WH_half, IoW_half, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,
                                               M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)

    #----------------------------------------- Step 4 : Update psi_i at time t^{n+1}: -----------------------------------------#
    A_hat, L_hat = YL_solvers.WF_hat_psi_SV(dim, n_z, Lw, H0, dt, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                                            dWM_dt, Uu, Ww, VoW, IoW,
                                            M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)
"""
    **************************************************************************************
    *                                 Define the solvers                                 *
    ************************************************************************************** """

#____________________________________ Solvers parameters ____________________________________#
'''
param_h       = {'ksp_converged_reason':None, 'pc_type': 'fieldsplit','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'}               
param_psi     = {'ksp_converged_reason':None, 'ksp_type': 'preonly', 'pc_type': 'lu'}
param_hat_psi = {'ksp_converged_reason':None, 'ksp_type': 'preonly', 'pc_type': 'lu'}
'''
# before optimisation
param_h       = {'ksp_converged_reason':None}               
param_psi     = {'ksp_converged_reason':None}
param_hat_psi = {'ksp_converged_reason':None}

# By default, the solve call will use GMRES using an incomplete LU factorisation to precondition the problem.
# We may solve the system directly by computing an LU factorisation of the problem. 
# To do this, we set the pc_type to 'lu' and tell PETSc to use a “preconditioner only” Krylov method.
# see https://www.firedrakeproject.org/solving-interface.html#solving-linear-systems
# 'ksp_view': None => print the solver parameters
# 'snes_view': None => print the residual
# 'snes_monitor': None

#--------------------------------------------------------------------------------------------#
#                                      Symplectic-Euler                                      #
#____________________________________________________________________________________________#
if scheme=="SE":
    #_______________________ Variational solver for h (and hat_psi^*) _______________________#
    h_problem = NonlinearVariationalProblem(WF_h_psi, w_n1)
    h_solver = NonlinearVariationalSolver(h_problem, options_prefix="h_dt_imp", solver_parameters=param_h)

    #_____________________________ Variational solver for psi_1 _____________________________#
    psi_problem = LinearVariationalProblem(A_psi_s, L_psi_s, psi_1_n1)
    # yl comment:
    # In this linear solver the trial function is psi_1.
    # psi_1_n1 is a function holding the solution, or we place the solution in psi_1_n1.
    psi_solver = LinearVariationalSolver(psi_problem, options_prefix="psi1_dt_exp", solver_parameters=param_psi)

    #____________________________ Variational solver for hat_psi ____________________________#
    hat_psi_problem = LinearVariationalProblem(A_hat, L_hat, hat_psi_n0)
    # yl comment:
    # psi_1_n1 was created and initialised (for calculating energy at t=0), in this linear solver the trial function is hat_psi.
    hat_psi_solver = LinearVariationalSolver(hat_psi_problem, options_prefix="hat_psi_exp", solver_parameters=param_hat_psi)

#--------------------------------------------------------------------------------------------#
#                                       Stormer-Verlet                                       #
#____________________________________________________________________________________________#
if scheme=="SV":
    #_______________________ Variational solver for psi_1^{n+1/2} (and hat_psi^*) _______________________#
    psi_half_problem = NonlinearVariationalProblem(WF_psi_star, w_half)
    psi_half_solver = NonlinearVariationalSolver(psi_half_problem, options_prefix="psi1_dt2_imp")
    
    #____________________________ Variational solver for h^{n+1} (and hat_psi^**) _______________________#
    h_problem = NonlinearVariationalProblem(WF_h_psi, w_n1)
    h_solver = NonlinearVariationalSolver(h_problem, options_prefix="h_dt_imp")
    
    #_______________________ Variational solver for psi_1^{n+1} _______________________#
    psi_n1_problem = LinearVariationalProblem(a_psi_1, L_psi_1, psi_1_n1)
    psi_n1_solver = LinearVariationalSolver(psi_n1_problem, options_prefix="psi1_dt_exp")
    
    #____________________________ Variational solver for hat_psi ____________________________#
    hat_psi_problem = LinearVariationalProblem(A_hat, L_hat, hat_psi_n0)
    hat_psi_solver = LinearVariationalSolver(hat_psi_problem, options_prefix="hat_psi_exp")

PETSc.Sys.Print('...solvers initialised!')

"""
    *************************************************************
    *                        Saving Files                       *
    ************************************************************* """
readme_file = os.path.join(save_path, 'readme.txt')

if case in ['TC1', 'TC2']: # save the energy file separately
    energy_file = os.path.join(save_path, 'energy.csv')
    energy_data = np.empty((0,2)) # create an empty 2d array for saving E(t)
    format_e = '%10.4f '+' %.18e '
    with open(energy_file,'w') as e_f:
        np.savetxt(e_f, energy_data, fmt=format_e)

elif case=='TC3':
    check_file = os.path.join(save_path, 'checkpoints.csv')
    check_data = np.empty((0,10))
    format_c = '%10.4f '+' %.18e '*9
    if hor_mesh.comm.rank==0:
        with open(check_file,'w') as c_f:
            np.savetxt(c_f, check_data, fmt=format_c)

elif case=='TC4':
    probe_file = os.path.join(save_path, 'probes.csv')
    probe_data = np.empty((0,10))
    format_p = '%10.4f'*9 + ' %.18e'
    if hor_mesh.comm.rank==0:
        with open(probe_file,'w') as p_f:
            np.savetxt(p_f, probe_data, fmt=format_p)

"""
    ****************************************************************************
    *                                 Saving mesh                              *
    ****************************************************************************"""
#---------------------------------------------------------------------------------#
#                      Save waves in the 3D free-surface domain                   #
#---------------------------------------------------------------------------------#
'''
if dim=='2D': # Extend the 1D horizontal mesh (x) to 2D horizontal mesh (x,y)
    mesh_2D = RectangleMesh(Nx,1,Lx,Ly,quadrilateral=True)        # 2D surface mesh
    V_2D = FunctionSpace(mesh_2D,"CG",1)                  # 2D surface funct. space
    Vec_2D = VectorFunctionSpace(mesh_2D,"CG",1, dim=n_z)  # 2D vector funct. space
    h_2D = Function(V_2D)                                                  # h(x,y)
    psi_s_2D = Function(V_2D)                                         # psi_1 (x,y)
    psi_i_2D = Function(Vec_2D)                                       # psi_i (x,y)
    # yl updated:
    WM_2D = Function(V_2D)
    x2 = SpatialCoordinate(mesh_2D)                                       
    beach_s_2D = Function(V_2D).interpolate(conditional(le(x2[0],xb),0.0,sb*(x2[0]-xb)))
    # Extend the surface mesh in depth to obtain {0<x<Lx; 0<y<Ly; 0<z<H0}
    mesh_3D = ExtrudedMesh(mesh_2D,                   # horizontal mesh to extrude;
                           n_z,               # number of elements in the vertical;
                           layer_height=H0/(n_z),         # length of each element;
                           extrusion_type='uniform')     # type of extruded coord.;

else:# If the solutions are already (x,y)-dependent, we extend the domain in depth:
    mesh_3D = ExtrudedMesh(hor_mesh,                  # horizontal mesh to extrude;
                           n_z,               # number of elements in the vertical;
                           layer_height=H0/(n_z),         # length of each element;
                           extrusion_type='uniform')     # type of extruded coord.;

"""
    *****************************
    *      Function to save     *
    ***************************** """
#__________ Function Space _________#
V_3D = FunctionSpace(mesh_3D, "CG",1)
#____________ Functions ____________#
waves = Function(V_3D,name= "phi")  # yl notes: store all the phi(x,y,z,t^n0) in the domain.
WM_3D = Function(V_3D,name = "WM") 


"""
    **************************************************************************
    *                         Mapping and transforms                         *
    **************************************************************************"""
if dim=="2D":
    # Indices to map h(x) and phi(x) to h(x,y) and phi(x,y) :
    Indx = []
    nodes=len(hor_mesh.coordinates.dat.data)
    if nodes<=1001:
        for j in range(nodes):
            Indx.append([y for y in range(len(mesh_2D.coordinates.dat.data[:,0]))\
            if mesh_2D.coordinates.dat.data[y,0]==hor_mesh.coordinates.dat.data[j]])
    else:
        for j in range(nodes-2):
            Indx.append([2*(nodes-1-j),2*(nodes-1-j)+1])
        Indx.append([0,1])
        Indx.append([2,3])
# yl notes: len(hor_mesh.coordinates.dat.data) = Nx+1, len(Indx) = Nx+1, each element is a pair of numbers.
# For the jth set Indx[j], each element represents the No. of the node on the vertical line x=(j-1)*(Lx/Nx),j=1,2,...,Nx+1

# Index used to differentiate each vertical layer
Indz = []
for i in range(0,n_z+1):
    Indz.append([zz for zz in range(len(mesh_3D.coordinates.dat.data[:,2])) \
     if mesh_3D.coordinates.dat.data[zz,2] == mesh_3D.coordinates.dat.data[i,2]])
# yl notes: len(Indz)=n_z+1, each element is also a list (set) with (Nx+1)*(Ny+1) numbers.
# For the ith set Indz[i], each element represents the No. of the node on the plane z=(i-1)*(H0/n_z),i=1,2,...,n_z+1

# Index of the 3D funct. for which x<Lw. This is used to transform the 3D domain
# in x, to get back to the moving domain:
Test_x_Lw=Function(V_3D)
# yl updates:
x1 = SpatialCoordinate(mesh_3D) # Need to define the coordinates again. yl.
Test_x_Lw.interpolate(conditional(le(x1[0],Lw),1.0,0.0))

Indw = [item for item in range(len(Test_x_Lw.dat.data[:])) if Test_x_Lw.dat.data[item] != 0.0]
# yl notes: the set of nodes located before x=Lw. Each element represents the No. of that node.
'''
PETSc.Sys.Print('Update of the solutions:')
""" *********************************************************************************
    *                                   Time loop                                   *
    ********************************************************************************* """
t_save = t+dt_save # do not save at t=T0
before_it = time.perf_counter()-start_time # running time from start until this line
smallfac = 10.0**(-10)  #  Onno 13-04-2020

#pdb.set_trace()

while t<=Tend+smallfac: #  while t<Tend-dt: Onno 15-04-2020
    """ *****************************************************************************
        *                               SAVE FUNCTIONS                              *
        ***************************************************************************** """
    if t_save-smallfac < t: # Onno 13-04-2020
        # yl updated:
        progress = format(100*(t-T0)/(Tend-T0), '.3f')+' %'
        tt = format(t, '.4f')
        PETSc.Sys.Print('t= %s, Progress: %s' % (tt, progress))
        #or PETSc.Sys.Print('Progress(%%): ', 100*t/Tend) or PETSc.Sys.Print('Progress', 100*t/Tend,' %%')
        #print_memory_usage(" - > ")
        #-------------------------------------------------------------------------------#
        #                                    ENERGY                                     #
        #-------------------------------------------------------------------------------#
        # yl update: XUVW
        tot_energy = save_energy(dim, Lw, H0, g, H, h_n0, psi_1_n0, hat_psi_n0, Uu, Ww, VoW, IoW, WH, b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,  
                                 A11, AN1, A1N, ANN, M11, M1N, MN1, MNN, D11, D1N, DN1, DNN, S11, S1N, SN1, SNN, I1, IN)
        
        if case in ['TC1', 'TC2']:
            energy_data = np.array([[t,tot_energy]])
            with open(energy_file,'a') as e_f:
                np.savetxt(e_f, energy_data, fmt=format_e)
            
            # Store the field data to a numpy array
            field_data = np.empty((Nx+1,4))
            for row_i in range(Nx+1):
                x_i = row_i*(Lx/Nx) # x coordinates
                field_data[row_i,:]=np.array([ x_i, h_n0.at(x_i), psi_1_n0.at(x_i), hat_psi_n0.at(x_i)[-1] ])

            # Save the array to a binary file (.npy)
            op_file = os.path.join(save_path, tt+'.npy')
            with open(op_file,'wb') as f:
                np.save(f, field_data)

        elif case=='TC3': # To compare with the old results where z_i's are evenly distributed
            check_data = np.array([[t, tot_energy, *h_n0.at([0,0], [Lx,0], [Lx,Ly]),\
                         WM.at([0,0]), dWM_dy.at([0,0]), dWM_dt.at([0,0]), WM.at([0,Ly]), dWM_dt.at([0,Ly])]])
            if hor_mesh.comm.rank==0:
                with open(check_file,'a') as c_f:
                    np.savetxt(c_f, check_data, fmt=format_c)

        elif case=='TC4':
            probe_data = np.array([[t,*h_n0.at(10, 20, 40, 49.5, 50, 54),WM.at(0),dWM_dt.at(0), tot_energy]])
            if hor_mesh.comm.rank==0:
                with open(probe_file,'a') as p_f:
                    np.savetxt(p_f, probe_data, fmt=format_p)

        '''
        if case=='TC2': #--Output h(x,y,t) for TC2 and TC2b--#
            h_file_name = 'h_'+tt+'.txt'
            h_file = open(os.path.join(save_path, h_file_name), 'w')
            y_coarse = np.linspace(0,Ly,17)
            x_coarse = np.linspace(0,Lx,81)
            for ix in x_coarse:
                for iy in y_coarse:
                    h_file.write('%-25s %-25s %-25s\n' %(str(ix), str(iy), str(h_n0.at(ix,iy))))
            h_file.close() 
        '''

        #-------------------------------------------------------------------------------#
        #                               SAVE 3D FUNCTIONS                               #
        #-------------------------------------------------------------------------------#
        '''
        if save_pvd:
            #______________________________ Project solutions ______________________________#
            if dim == '2D':
                # To the surface plane (x,y): # add WM by yl.
                x_to_xy(h_n0, WM, psi_1_n0, hat_psi_n0, h_2D, WM_2D, psi_s_2D, psi_i_2D, Indx)
                # In depth (x,y,z):
                for i in range(0,n_z+1):                                     # for each layer
                    phi_projection(i, n_z, waves, Indz, psi_s_2D, psi_i_2D)  # phi(z) = psi_i
                    # WM_3D.dat.data[Indz[i]] = WM.dat.data[0]                 # WM(z) = WM. 
                    # yl: Wrong! will cause mistake in x_transformation
                    # yl updated:
                    WM_3D.dat.data[Indz[i]] = WM_2D.dat.data[:]
            elif dim == '3D':
                # In depth (x,y,z):
                for i in range(0,n_z+1):                                     # for each layer
                    phi_projection(i, n_z, waves, Indz, psi_1_n0, hat_psi_n0)# phi(z) = psi_i
                    WM_3D.dat.data[Indz[i]] = WM.dat.data[:]                     # WM(z) = WM

            #__________________________ Save the fixed coordinates _________________________#
            init_coord = mesh_3D.coordinates.vector().get_local()

            #_________________________________ z-transform _________________________________#
            if dim == '2D':
                z_transform(mesh_3D, n_z, h_2D, beach_s_2D, H0, Indz)
            elif dim == '3D':
                z_transform(mesh_3D, n_z, h_n0, b, H0, Indz)

            #_________________________________ x-transform _________________________________#
            x_transform(mesh_3D, Lw, WM_3D, Indw)

            #_________________________________ Save waves __________________________________#
            save_waves.write(waves)
            # yl notes: now the waves values are saved to the time-dependent mesh.

            #__________________________ Back to the initial mesh ___________________________#
            mesh_3D.coordinates.vector().set_local(init_coord)
        
            #_______________________________ Save wavemaker ________________________________#
            save_WM.write(WM_3D)
            # yl notes: but the wave_maker values are saved to the time-independent mesh.
        '''
        #_____________________________ Update saving time ______________________________#
        t_save+=dt_save
        

    """ *********************************************************************
        *                            Update time                            *
        ********************************************************************* """

    #_______________________ Update time: t^n -> t^{n+1} _______________________#
    t_half = t+0.5*dt
    t += dt
    
    if input_data=="created":
        t_half_const.assign(t_half)
        t_const.assign(t)

        if scheme=="SV":
            # updated by yl
            WM_expr(WM_half, x, t_half_const, t_stop)                            # update R(x,y;t)
            dWM_dt_expr(dWM_half_dt, x, t_half_const, t_stop)                    # update dR/dt
            dWM_dy_expr(dWM_half_dy, x, t_half_const, t_stop)                    # update dR/dy
            # yl update: XUVW
            Uu_half.interpolate(Xx*dWM_half_dy)
            Ww_half.interpolate(Lw-WM_half)
            Vv_half.interpolate(Lw*Lw+Uu_half*Uu_half)
            VoW_half.interpolate(Vv_half/Ww_half)
            IoW_half.interpolate(1/Ww_half)
            WH_half.interpolate(Ww_half*H)
            XRt_half.interpolate(Xx*dWM_half_dt)

        # updated by yl
        WM_expr(WM_n1, x, t_const, t_stop)                            # update R(x,y;t)
        dWM_dt_expr(dWM_n1_dt, x, t_const, t_stop)                    # update dR/dt
        dWM_dy_expr(dWM_n1_dy, x, t_const, t_stop)                    # update dR/dy
        # yl update: XUVW
        Ww_n1.interpolate(Lw-WM_n1)

    else: # yl added. input_data=='measurements'
        
        if scheme=="SV":
            Rt = round(np.interp(t_half, t_wm, wm_data), 4)
            Rt_const.assign(Rt)
            Rt_t = round(np.interp(t_half, t_vel, vel_data), 4)
            Rt_t_const.assign(Rt_t)
            WM_expr(WM_half,x,Rt_const)
            dWM_dt_expr(dWM_half_dt,x,Rt_t_const)                    
            # yl update: XUVW
            Ww_half.interpolate(Lw-WM_half)
            VoW_half.interpolate(Vv_half/Ww_half)
            IoW_half.interpolate(1/Ww_half)
            WH_half.interpolate(Ww_half*H)
            XRt_half.interpolate(Xx*dWM_half_dt)

        Rt = round(np.interp(t, t_wm, wm_data), 4)
        Rt_const.assign(Rt)
        Rt_t = round(np.interp(t, t_vel, vel_data), 4)
        Rt_t_const.assign(Rt_t)
        WM_expr(WM_n1,x,Rt_const)
        dWM_dt_expr(dWM_n1_dt,x,Rt_t_const)
        # yl update: XUVW
        Ww_n1.interpolate(Lw-WM_n1)

    """ **************************************************
        *            Solve the weak formulations         *
        ************************************************** """
    #___________________ Call the solvers ___________________#
    if scheme=="SE":                     # 1st-order SE scheme
        h_solver.solve()           # get h^{n+1} and hat_psi^*
        psi_solver.solve()                     # get psi^{n+1}
    elif scheme=="SV":                   # 2nd-order SV scheme
        psi_half_solver.solve() # get psi^{n+1/2} and hat_psi^*
        h_solver.solve()        # get h^{n+1} and hat_psi^{**}
        psi_n1_solver.solve()                  # get psi^{n+1}
    
    """ *************************************************
        *               Update the functions            *
        ************************************************* """
    #_________________ Update the solutions ________________#
    h_out, hat_psi_out = w_n1.subfunctions
    h_n0.assign(h_out)
    psi_1_n0.assign(psi_1_n1)
    hat_psi_n0.assign(hat_psi_out)
    
    #_________________ Update the wavemaker ________________#
    WM.assign(WM_n1)
    dWM_dt.assign(dWM_n1_dt)
    dWM_dy.assign(dWM_n1_dy)

    # yl update: XUVW
    Uu.interpolate(Xx*dWM_dy)
    Ww.interpolate(Lw-WM)
    Vv.interpolate(Lw*Lw+Uu*Uu)
    VoW.interpolate(Vv/Ww)
    IoW.interpolate(1/Ww)
    WH.interpolate(Ww*H)
    XRt.interpolate(Xx*dWM_dt)

comp_time = time.perf_counter()-start_time
jours = int(comp_time/(24*3600))
heures = int((comp_time-jours*24*3600)/3600)
minutes = int((comp_time-jours*24*3600-heures*3600)/60)
secondes = comp_time -jours*24*3600-heures*3600 - minutes*60
with open(readme_file,'w') as info:
    save_README(info, dim, Lx, Ly, H0, xb, sb, Nx, Ny, n_z, gamma, Tw, w, t_stop, Lw, scheme, dt, t,\
                jours, heures, minutes, secondes, comp_time, before_it,COMM_WORLD.size)
