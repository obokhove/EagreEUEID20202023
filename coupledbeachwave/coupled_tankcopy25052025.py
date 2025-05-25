# 2D Coupled Numerical Wavetank code adapted from "3D_tank.py".
# Currently can only run in serial. 
# Needs to find a "gather" function to assamble data from all ranks at rank 0 for FV.

import pdb
import time
import numpy as np
import os.path

# monitor memory usage 
import psutil, os
def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024**2  # Resident Set Size in MB
    print(f"{label} Memory usage = {mem_mb:.2f} MB")

import sympy as sp
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule

from firedrake import *
from firedrake.petsc import PETSc
import dw_fe_solver as YL_solvers
from sw_fv_solver import *
from savings import *
from settings import *
# from memory_profiler import profile #  ONNO added/removed

start_time = time.perf_counter()

"""
    ****************************************
    *               Settings               *
    **************************************** """
PETSc.Sys.Print('Setting up coupled wavetank...')

input_data, test_case, FWF, save_pvd = test_case()

#___________________ Deep-water domain ____________________#
H0, xb, slope, Hc, Ldw, Lw, res_dw, n_z = domain()

#__________________ Sallow-water domain ___________________#
Lsw = (1.1*H0 - (H0-Hc))/slope # 10% margin
res_sw = res_dw*res_dw

#______________________ Whole domain ______________________#
L_total = Ldw + Lsw

#_______________________ Wavemaker ________________________#
g, lamb, k, w, Tw, gamma, t_stop = wavemaker(H0, Lw)

#__________________________ Time __________________________#
T0, t, dt, Tend, dt_save = set_time(Tw)

PETSc.Sys.Print('...settings loaded!')

# Create the directory to store output files
save_path = 'data/' + test_case
try:
    os.makedirs(save_path, exist_ok = True)
    PETSc.Sys.Print("Directory created successfully!")
except OSError as error:
    PETSc.Sys.Print("Directory can not be created!")

nonno = 0 # ONNO added switch as VTKFile does not work
if nonno==0:
    from firedrake import VTKFile #  ONNO added extra
    dw_beach_file = VTKFile(os.path.join(save_path, "dw_beach.pvd"))
    dw_h_file     = VTKFile(os.path.join(save_path, "dw_h_t.pvd"))
    sw_beach_file = VTKFile(os.path.join(save_path, "sw_beach.pvd"))
    sw_h_file     = VTKFile(os.path.join(save_path, "sw_h_t.pvd"))
    dw_waves_file = VTKFile(os.path.join(save_path, "dw_waves.pvd"))
    sw_waves_file = VTKFile(os.path.join(save_path, "sw_waves.pvd"))
else:
    dw_beach_file = File(os.path.join(save_path, "dw_beach.pvd"))
    dw_h_file     = File(os.path.join(save_path, "dw_h_t.pvd"))
    sw_beach_file = File(os.path.join(save_path, "sw_beach.pvd"))
    sw_h_file     = File(os.path.join(save_path, "sw_h_t.pvd"))
    dw_waves_file = File(os.path.join(save_path, "dw_waves.pvd"))
    sw_waves_file = File(os.path.join(save_path, "sw_waves.pvd"))

"""
    ****************************************************
    *               Definition of the mesh             *
    **************************************************** """
PETSc.Sys.Print('Creation of the mesh across %d processes...' % COMM_WORLD.size)

#_________________ Vertical discretization ________________#
Nz = n_z+1         # Number of point in one vertical element

#________________ Horizontal discretization _______________#
Ne_dw = round(Ldw/res_dw)    # Number of elements in DW (round to the nearest integer)
Nv_sw = round(Lsw/res_sw)    # Number of volumns in SW

#____________________ Deep-water mesh _____________________#
# Generate a uniform mesh of an interval
dw_mesh = IntervalMesh(Ne_dw,Ldw) 

#___________________ Shallow-water mesh ___________________#
sw_mesh = IntervalMesh(Nv_sw,Ldw,L_total)

x = SpatialCoordinate(dw_mesh)
x_sw = SpatialCoordinate(sw_mesh)

PETSc.Sys.Print('...mesh created!')


"""
    *************************************************
    *       Definition of the function spaces       *
    ************************************************* """
PETSc.Sys.Print('Definition of the functions...')

#_______________ Deep-water function space ________________#
dw_V = FunctionSpace(dw_mesh, "CG", 1)
dw_Vec = VectorFunctionSpace(dw_mesh, "CG", 1, dim=n_z)
dw_W = dw_V * dw_Vec

#______________ Shallow-water function space ______________#
sw_V = FunctionSpace(sw_mesh, "DG", 0)

"""
    ******************************************************
    *            Definition of the functions             *
    ****************************************************** """
#================== Deep-water functions ==================#
#_______________________ At time t^n ______________________#
h_n0 = Function(dw_V)                                  # h^n
psi_1_n0 = Function(dw_V)                          # psi_1^n
hat_psi_n0 = Function(dw_Vec)                    # hat_psi^n
WM = Function(dw_V)                               # R(x,t^n)
dWM_dt = Function(dw_V)                          # (dR/dt)^n

#_________________ At time t^{n+1} and t^* ________________#
w_n1 = Function(dw_W)
h_n1, hat_psi_star = split(w_n1)        # h^{n+1}, hat_psi^*
psi_1_n1 = Function(dw_V)                      # psi_1^{n+1}
hat_psi_n1 = Function(dw_Vec)   # to visualise hat_psi^{n+1}
WM_n1 = Function(dw_V)                        # R(x,t^{n+1})
dWM_n1_dt = Function(dw_V)                   # (dR/dt)^{n+1}

#_________ Bottom topography in deep-water region _________#
dw_b = Function(dw_V, name="beach_dw")                # b(x)

#____________ Rest depth in deep-water region _____________#
H = Function(dw_V)                                    # H(x)

#_____________________ Trial functions ____________________#
psi_1 = TrialFunction(dw_V)        # for solving psi_1^{n+1}
hat_psi = TrialFunction(dw_Vec)  # for solving hat_psi^{n+1}

#______________________ Test functions ____________________#
delta_h = TestFunction(dw_V)                         
delta_hat_psi = TestFunction(dw_Vec)
w_t = TestFunction(dw_W)
delta_psi, delta_hat_star = split(w_t)

#____________________ yl update: XUVW _____________________#
Xx = Function(dw_V)                          # X(x) = x - Lw
Ww = Function(dw_V)         # W(x,t^n) = Lw - tilde_R(x,t^n)
Ww_n1 = Function(dw_V)      # W(x,t^{n+1})
IoW = Function(dw_V)        # 1/W

#================ Shallow-water functions =================#
h_fv = Function(sw_V)
hu_fv = Function(sw_V)
hu_fe = Function(dw_V) # Used for the boundary terms in the deep-water weak forms

#_______ Bottom topography in shallow-water region ________#
sw_b = Function(sw_V)                           # check_b(x)

#=============== Visualisation in ParaView ================#
dw_h = Function(dw_V, name="h(t)_dw")
sw_h = Function(sw_V, name="h(t)_sw")
sw_b_out = Function(sw_V, name="beach_sw")

PETSc.Sys.Print('...functions created!')

"""
    ***********************************************************************************
    *                          Initialisation of the Functions                        *
    ***********************************************************************************"""
PETSc.Sys.Print('Initalisation of the functions...')
#----------------------------------------------------------------------------------------#
#                                       Topography                                       #
#----------------------------------------------------------------------------------------#
#_________________ Deep-water topography __________________#
H.interpolate( H0-conditional(le(x[0],xb),0.0,slope*(x[0]-xb)) ) # Water depth at rest H(x)
dw_b.assign(H0-H)                          # Bathymetry b(x)
dw_beach_file.write(dw_b)                      # save data

#________________ Shallow-water topography ________________#
sw_b.interpolate(slope*(x_sw[0]-Ldw))
H0_sw = Hc - sw_b.dat.data[0]  # yl:note that b(x) is non-zero at x_c due to FV
sw_b_out.assign(sw_b + dw_b.dat.data[-1])
sw_beach_file.write(sw_b_out)

# for the FV solver
bk = np.zeros((1, Nv_sw + 1))
bk[0,1:Nv_sw+1] = sw_b.dat.data[:]
bk[0,0] = bk[0,1]-(bk[0,2]-bk[0,1])
bk[0,-1] = bk[0,-2]
bk_half_r = np.maximum(bk[0, 1:Nv_sw],bk[0, 2:Nv_sw+1])    # b_{k+1/2}
bk_half_l = np.maximum(bk[0, 1:Nv_sw],bk[0, 0:Nv_sw-1])    # b_{k-1/2}

#----------------------------------------------------------------------------------------#
#                                       Wavemaker                                        #
#----------------------------------------------------------------------------------------#
t_const = Constant(T0)
R_expr = conditional(le(x[0], Lw), -gamma * cos(w * t_const), 0.0)
dRdt_expr = conditional(le(x[0], Lw), gamma * w * sin(w * t_const), 0.0)

WM.interpolate(R_expr)                 # \tilde{R}(x,t)
dWM_dt.interpolate(dRdt_expr)          # d\tilde{R}/dt

#----------------------------------------------------------------------------------------#
#                               Solutions Initialization                                 #
#----------------------------------------------------------------------------------------#
#______________________________________ Deep water ______________________________________#
h_n0.assign(H)        # h(x,y;t=0) = H(x)                                                  
w_n1.sub(0).assign(H) # Extract the ith sub Function of this Function. In this case, h^{n+1}.
#_____________________________________ Shallow water ____________________________________#
h_fv.assign(H0_sw - sw_b) # yl: note that H0_sw is used rather than Hc
# FG: h_fv.dat.data[np.where(h_fv.vector().get_local()<0)]=0
h_fv.dat.data[np.where(h_fv.dat.data < 0)] = 0
H_sw = np.copy(h_fv.dat.data) # for calculating shallow-water energy

# yl update: XUVW
Xx.interpolate(x[0]-Lw)
Ww.assign(Lw-WM)
IoW.interpolate(1/Ww)

PETSc.Sys.Print('...functions initialised!')

"""
    ************************
    * Compute the matrices *
    ************************ """
PETSc.Sys.Print('Assembling z-matrices...')
#_______ Initialization ______#
A = np.zeros((Nz,Nz))
B = np.zeros((Nz,Nz)) # FWF
C = np.zeros((Nz,Nz)) # FWF
M = np.zeros((Nz,Nz))
D = np.zeros((Nz,Nz))
S = np.zeros((Nz,Nz))
Ik = np.zeros((Nz,1))
Gk = np.zeros((Nz,1)) # for coupling BC

Jk = np.zeros((Nz,1)) # Jk[i] = sp.integrate(sp.diff(varphi[i],z),(z,0,H0))
Jk[0,0] = 1
Jk[-1,0] = -1

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
    Gk[i] = sp.integrate(z*sp.diff(varphi[i],z),(z,0,H0))

#________ Submatrices ________#
# Note: np.array_equal(np.array(XN1), np.array(X1N))
# 'True'  for A, M, S, C
# 'False' for D, B

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

S11 = S[0,0]
S1N = as_tensor(S[0,1:])
SN1 = as_tensor(S[1:,0])
SNN = as_tensor(S[1:,1:])

I1 = Ik[0,0]
IN = as_tensor(Ik[1:,0])

G1 = Gk[0,0] # G1 = -I1 + H_0
GN = as_tensor(Gk[1:,0]) # GN = -IN

J1 = Jk[0,0]
JN = as_tensor(Jk[1:,0])

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

"""
    ************************************************************************************************************************
    *                                                   Weak Formulations                                                  *
    ************************************************************************************************************************ """
PETSc.Sys.Print('Initialisation of the solvers...')
#_____________________________________________ Symplectic-Euler ______________________________________________#
#------------------------ Step 1 : Update h at time t^{n+1} and psi_i at time t^* simulataneously: ------------------------#
WF_h_psi = YL_solvers.WF_h_SE(n_z, g, H0, Lw, dWM_dt, dt, delta_psi, delta_hat_star, h_n0, h_n1, psi_1_n0, hat_psi_star,
                              Xx, Ww, IoW, dw_b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF, hu_fe,
                              M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)

#----------------------------------------- Step 2 : Update psi_1 at time t^{n+1}: -----------------------------------------#
A_psi_s, L_psi_s = YL_solvers.WF_psi_SE(g, H0, H, Lw, dWM_dt, dt, delta_h, psi_1, psi_1_n0, hat_psi_star, h_n1, 
                                        Xx, Ww, Ww_n1, IoW, dw_b, C11, CN1, CNN, FWF, hu_fe,
                                        M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN, G1, GN)

#----------------------------------------- Step 3 : Update psi_i at time t^{n+1}: -----------------------------------------#
A_hat, L_hat = YL_solvers.WF_hat_psi_SE(H0, n_z, Lw, dWM_dt, dt, delta_hat_psi, hat_psi, h_n0, psi_1_n0, 
                                        Ww, IoW, dw_b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF, hu_fe,
                                        M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, A11, AN1, ANN, I1, IN)

"""
    **************************************************************************************
    *                                 Define the solvers                                 *
    ************************************************************************************** """

#--------------------------------------------------------------------------------------------#
#                                      Symplectic-Euler                                      #
#____________________________________________________________________________________________#

#_______________________ Variational solver for h (and hat_psi^*) _______________________#
h_problem = NonlinearVariationalProblem(WF_h_psi, w_n1)
DW_solver_h = NonlinearVariationalSolver(h_problem)

#_____________________________ Variational solver for psi_1 _____________________________#
psi_problem = LinearVariationalProblem(A_psi_s, L_psi_s, psi_1_n1)
DW_solver_psi_s = LinearVariationalSolver(psi_problem)

#____________________________ Variational solver for hat_psi ____________________________#
hat_psi_problem = LinearVariationalProblem(A_hat, L_hat, hat_psi_n0)
DW_solver_hat_psi_n1 = LinearVariationalSolver(hat_psi_problem)

PETSc.Sys.Print('...solvers initialised!')

"""
    *************************************************************
    *                    Output Text Files                      *
    ************************************************************* """
readme_file = os.path.join(save_path, 'readme.txt')
# save the energy file separately
energy_file = os.path.join(save_path, 'energy.csv')

"""
    ****************************************************************************
    *                    WAVE VISUALISATION IN X-Z PlANE (YL)                  *
    ****************************************************************************"""

#============================== Deep Water Region ==============================#
zk_distances = [abs(z_k[i+1] - z_k[i]) for i in range(len(z_k) - 1)]
dw_mesh_2D = ExtrudedMesh(dw_mesh, n_z, layer_height = zk_distances, extrusion_type = 'uniform')
dw_V_2D  = FunctionSpace(dw_mesh_2D, "CG", 1) # the x and z parts have the same family and degree

dw_waves = Function(dw_V_2D, name="phi_dw")
WM_2D    = Function(dw_V_2D)

init_dw_coord = dw_mesh_2D.coordinates.vector().get_local()

#============================ Shallow Water Region =============================#
sw_mesh_2D = ExtrudedMesh(sw_mesh, 1, layer_height = H0, extrusion_type = 'uniform')
sw_V_2D = FunctionSpace(sw_mesh_2D, "DG", 0)
sw_waves = Function(sw_V_2D, name="u_sw")
# note that sw_waves.dat.data.shape=(Nv,)

init_sw_coord = sw_mesh_2D.coordinates.vector().get_local()

"""
    **************************************************************************
    *                         Mapping and transforms                         *
    **************************************************************************"""

#___________ Index used to differentiate each vertical layer (1D ->2D) _________#
#--------------------------------- Deep water: ---------------------------------#
tolerance = 1e-8  # Small tolerance for floating-point comparison
Indz_dw = []
for i in range(0, n_z+1):
    layer_indices = np.where(np.abs(dw_mesh_2D.coordinates.dat.data[:, 1] - dw_mesh_2D.coordinates.dat.data[i, 1]) < tolerance)
    Indz_dw.append(layer_indices)

# Indz_dw[0] --> z=0; ...; Indz_dw[n_z] --> z=H0

#-------------------------------- Shallow water: -------------------------------#
Indx_sw = []
for i in range(len(sw_mesh.coordinates.dat.data)):
    Indx_sw.append(np.where(sw_mesh_2D.coordinates.dat.data[:,0]==sw_mesh.coordinates.dat.data[i]))
# find the indexes for nodes at the right end of the mesh
sw_r_end = np.where(sw_mesh_2D.coordinates.dat.data[:,0]==np.max(sw_mesh_2D.coordinates.dat.data[:,0]))

""" *********************************************************************************
    *                                   Time loop                                   *
    ********************************************************************************* """
PETSc.Sys.Print('Update of the solutions:')
t_save = t
#t_save = t+dt_save # Do NOT save from T0 as energy difference is computed from n=1
before_it = time.perf_counter()-start_time # before iteration/time loop
smallfac = 10.0**(-10)

#pdb.set_trace()
while t<=Tend+smallfac:
    """ *****************************************************************************
        *                               SAVE FUNCTIONS                              *
        ***************************************************************************** """

    if t_save-smallfac < t:
        progress = format(100*(t-T0)/(Tend-T0), '.3f')+' %'
        tt = format(t, '.4f')
        PETSc.Sys.Print('t= %s, Progress: %s' % (tt, progress))
        print_memory_usage(" - > ")
        #-------------------------------------------------------------------------------#
        #                                    ENERGY                                     #
        #-------------------------------------------------------------------------------#
        E_dw = dw_energy(Lw, H0, g, H, h_n0, psi_1_n0, hat_psi_n0, 
                         Ww, IoW, dw_b, C11, CN1, CNN, B11, B1N, BN1, BNN, FWF,  
                         A11, AN1, ANN, M11, MN1, MNN, D11, D1N, DN1, DNN, S11, SN1, SNN, I1, IN)
        
        E_sw = sw_energy(Nv_sw, res_sw, g, H_sw, h_fv, hu_fv)
        
        #_________________________ Find the waterline at x=x_w _________________________#
        zero_index = np.where(h_fv.dat.data == 0)[0][0]
        x_w = sw_mesh.coordinates.dat.data[zero_index] # ONNO incorrect not the waterline
        z1_index = np.where(h_fv.dat.data < 1e-1)[0][0]
        x_w1 = sw_mesh.coordinates.dat.data[z1_index] # ONNO test
        z2_index = np.where(h_fv.dat.data < 1e-2)[0][0]
        x_w2 = sw_mesh.coordinates.dat.data[z2_index] # ONNO test
        z3_index = np.where(h_fv.dat.data < 1e-3)[0][0]
        x_w3 = sw_mesh.coordinates.dat.data[z3_index] # ONNO test
        z4_index = np.where(h_fv.dat.data < 1e-4)[0][0]
        x_w4 = sw_mesh.coordinates.dat.data[z4_index] # ONNO test
        z5_index = np.where(h_fv.dat.data < 1e-5)[0][0]
        x_w5 = sw_mesh.coordinates.dat.data[z5_index] # ONNO test

        with open(energy_file, 'a') as e_f:
            e_f.write(f"{t:8.4f} {E_dw:.6e} {E_sw:.6e} {x_w:10.6f} {x_w1:10.6f} {x_w2:10.6f} {x_w3:10.6f} {x_w4:10.6f} {x_w5:10.6f}\n")

        #-------------------------------------------------------------------------------#
        #                              SAVE 2D WATER DEPTH                              #
        #-------------------------------------------------------------------------------#
        
        #__________________________________ Deep-Water _________________________________#
        dw_h.assign(h_n0 + dw_b)
        dw_h_file.write(dw_h, time = t)
        #_________________________________ Shallow-Water _______________________________#
        sw_h.assign(sw_b.dat.data[0]+ dw_b.dat.data[-1] + sw_b + h_fv)
        sw_h_file.write(sw_h, time = t)

        #-------------------------------------------------------------------------------#
        #                               SAVE 2D FUNCTIONS                               #
        #-------------------------------------------------------------------------------#
        
        #============================== Deep Water Region ==============================#
        #______________________________ Project solutions ______________________________#
        for i in range(0,n_z+1):
            dw_phi_projection(i, n_z, dw_waves, Indz_dw, psi_1_n0, hat_psi_n0)
            WM_2D.dat.data[Indz_dw[i]] = WM.dat.data[:]
        
        #_________________________________ z-transform _________________________________#
        dw_z_transform(dw_mesh_2D, n_z, h_n0, dw_b, H0, Indz_dw)
        
        #_________________________________ x-transform _________________________________#
        dw_x_transform(dw_mesh_2D, Lw, WM_2D)

        #_________________________________ Save waves __________________________________#
        dw_waves_file.write(dw_waves, time = t)

        #__________________________ Back to the initial mesh ___________________________#
        dw_mesh_2D.coordinates.vector().set_local(init_dw_coord)

        #============================ Shallow Water Region =============================#
        # assmbling sw_waves, note the reversed index order
        ind_wet = np.where(h_fv.dat.data > tolerance)
        rev_ind = ind_wet[::-1]
        sw_waves.dat.data[ind_wet] = hu_fv.dat.data[rev_ind]/h_fv.dat.data[rev_ind]

        # z-tranform
        for i in range(len(h_fv.dat.data)):
            sw_mesh_2D.coordinates.dat.data[Indx_sw[i],1] *= h_fv.dat.data[i]
            sw_mesh_2D.coordinates.dat.data[Indx_sw[i],1] += sw_b.dat.data[i]+ sw_b.dat.data[0]+ dw_b.dat.data[-1]
        # note that sw_mesh_2D.co.dat.data.shape=((Nv+1)*2,2), while h_fv.dat.data.shape=(Nv,), so right boundary:
        sw_mesh_2D.coordinates.dat.data[sw_r_end,1] = slope*(L_total-xb)
        # save waves
        sw_waves_file.write(sw_waves, time = t)
        #---- transform back
        sw_mesh_2D.coordinates.vector().set_local(init_sw_coord)
        
        #_____________________________ Update saving time ______________________________#
        t_save+=dt_save
        

    """ *********************************************************************
        *                            Update time                            *
        ********************************************************************* """
    
    t += dt
    t_const.assign(t)
    
    if t <= t_stop:
        WM_n1.interpolate(R_expr)
        dWM_n1_dt.interpolate(dRdt_expr)
    else:
        WM_n1.interpolate(conditional(le(x[0], Lw), -gamma * cos(w * t_stop), 0.0))
        dWM_n1_dt.assign(0.0)

    Ww_n1.assign(Lw-WM_n1)
    
    """ **************************************************
        *            Solve the weak formulations         *
        ************************************************** """
    #______________ Call the deep-water solvers ______________#
    DW_solver_h.solve()                      # h^{n+1}, psi_i^*
    DW_solver_psi_s.solve()                       # psi_1^{n+1}
    
    #__________ Update the boundary conditions for SW ________#
    h_out, hat_psi_out = w_n1.subfunctions
    
    h_bc = assemble((h_out)*ds(2))
    # yl updated: add b(x) term
    hu_bc = assemble((1/H0)*(  h_out*( psi_1_n0.dx(0)*I1 + dot(hat_psi_out.dx(0),IN) )\
                             - h_out.dx(0)*(psi_1_n0*G1 + dot(GN,hat_psi_out)) \
                           - FWF*H0*dw_b.dx(0)*(psi_1_n0*J1 + dot(JN,hat_psi_out)) )*ds(2))

    #______________ Call the shallow-water solver _____________#
    h_fv, hu_fv, hu_fe = solve_FV(Nv_sw, res_sw, dt, bk, bk_half_r, bk_half_l, g, 
                                  h_bc, hu_bc, h_fv, hu_fv, hu_fe)

    """ *************************************************
        *               Update the functions            *
        ************************************************* """
    #_________________ Update the solutions ________________#
    h_n0.assign(h_out)
    psi_1_n0.assign(psi_1_n1)
    hat_psi_n0.assign(hat_psi_out)
    
    #_________________ Update the wavemaker ________________#
    WM.assign(WM_n1)
    dWM_dt.assign(dWM_n1_dt)

    # yl update: XUVW
    Ww.assign(Lw-WM)
    IoW.interpolate(1/Ww)


comp_time = time.perf_counter()-start_time
jours = int(comp_time/(24*3600))
heures = int((comp_time-jours*24*3600)/3600)
minutes = int((comp_time-jours*24*3600-heures*3600)/60)
secondes = comp_time -jours*24*3600-heures*3600 - minutes*60
with open(readme_file,'w') as info:
    save_README(info, L_total, Ldw, Lsw, H0, xb, slope, Ne_dw, Nv_sw, n_z, gamma, Tw, w, t_stop, Lw, 
                dt, t, jours, heures, minutes, secondes, comp_time, before_it, COMM_WORLD.size)
