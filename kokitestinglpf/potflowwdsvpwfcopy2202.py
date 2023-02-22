import firedrake as fd
import math
import time
import numpy as np
import sympy as sp
from sympy import summation
from sympy.abc import k
import matplotlib.pyplot as plt
import os
# import os.path


# parameters in SI units
# t_end = 5.0  # time of simulation [s]
# dt = 0.005  # time step [s]
g = 9.81  # gravitational acceleration [m/s^2]

# water
Lx = 20.0  # length of the tank [m] in x-direction; needed for computing initial condition
Lz = 10.0  # height of the tank [m]; needed for computing initial condition
nx = 120 # a) 120 and b) 2*120 c) 2*2*120 d) 200
nz = 6 # a) 6 and b,c) 2*6 d) 

Lx = 140
Lz = 10
Lz = 1
nx = 420 #  Works for SV at 200, 400, 800 CFL = 0.125; works not well for SE;
nx = 280
nz = 6
#  nx = 280 nz = 4

H0 = Lz # rest water depth [m]


# control parameters
output_data_every_x_time_steps = 20  # to avoid saving data every time step

save_path =  "lin_pot_flow" 
if not os.path.exists(save_path):
    os.makedirs(save_path)                                       

top_id = 'top'

nvpcase = 233 # case 111 (SE linear), 2 (SE nonlinear), 21 (SE nonlinear wavemaker), 22 (SV nonlinear wavemaker) work and
# case 23 (midpoint NL wavemaker) ; 24 mmp wave-buoy does not converge; 25 mmp wave-waveflap (extension of 23 with mmp piston) is in progress;
# case 23 as case 233 has been slowly morphed to the waveflap case 25, the error within case 25 is still not found.

#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
tsize2 = 12
size = 16  # font size of image axes
factor = 0
t = 0
tt = format(t, '.3f') 

#________________________ MESH  _______________________#
# ONNO 05-01 change to extruded mesh; see example:
# https://www.firedrakeproject.org/demos/extruded_continuity.py.html
# https://www.firedrakeproject.org/extruded-meshes.html
# CG x R for surface eta and phi
# CG x CG for interior phi or varphi
# Use at for visualisation at point; use several at's.

mesh1d = fd.IntervalMesh(nx, Lx)
mesh = fd.ExtrudedMesh(mesh1d, nz, layer_height=Lz/nz, extrusion_type='uniform')

# mesh = fd.RectangleMesh(nx, nz, Lx, Lz)

x, z = fd.SpatialCoordinate(mesh)

xvals = np.linspace(0.0, Lx-10**(-10), nx)
zvals = np.linspace(0.0, Lz-10**(-10), nz) # 
zslice = H0
xslice = 0.5*Lx

# The equations are in nondimensional units, hence we 
L = 1
T = 1
# T t_end /= T
# Tdt /= T
Lx /= L
Lz /= L
gg = g

## initial condition nic=0 in fluid based on analytical solution
## compute analytical initial phi and eta a = 0.0 * T / L ** 2    # in nondim units b = 0.005 * T / L ** 2  # in nondim units

x = mesh.coordinates
## phi_exact_expr = a * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) # ONNO: Huh?
## eta_exact_expr = -omega * b * fd.cos(kx * x[0]) * fd.cosh(kx * Lz) # ONNO: Huh?

t0 = 0.0
nic = 0
if nvpcase == 21: # wavemaker
    nic = 1
elif nvpcase == 22: # wavemaker
    nic = 1
elif nvpcase == 23: # wavemaker
    nic = 1
elif nvpcase == 233: # wavemaker case 23 to 25 plus waveflap
    nic = 1
    nowaveflap = 1.0 # 0: pure piston case; 1: pure waveflap case
elif nvpcase == 24: # wavemaker
    nic = 1

time = []
t = 0
    
if nic == 0:
    n_mode = 2
    kx = 2 * np.pi * n_mode / Lx
    omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
    A = 0.2*4
    D = -gg*A/(omega*np.cosh(kx*H0))
    Tperiod = 2*np.pi/omega
    print('Period: ', Tperiod)
    phi_exact_expr = D * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
    phi_exact_exprH0 = D * fd.cos(kx * x[0]) * fd.cosh(kx * H0) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
    eta_exact_expr = A * fd.cos(kx * x[0]) * np.cos(omega * t0)
    t_end = Tperiod  # time of simulation [s]
    dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
    Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
    CFL = 0.125 # run at a) 0.125 and b) 0.5*0.125
    dt = CFL*Tperiod/Nt  # 0.005  # time step [s]
    print('dtt=',dtt, t_end/dtt,dt,2/omega)
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 4
elif nic == 1: 
    lambd = 70
    lambd = 10

    if nvpcase == 24:
        rhow = 1000 # density water
        Ly = 1 # lateral width
        MM = 125 # weight of buoy
        Slopebuoy = 1
        Keel = 0.5
        mm = MM/(rhow*Ly) # scaled weight of buoy
        Z0 = H0+Keel-np.sqrt(2.0*mm*Slopebuoy)  # Reference position buoy based on Archimedes law or hard constraint
        Lbb = Lx-(H0 + Keel - Z0)/Slopebuoy  # LxmLb = Lx-Lbb =H0-Keel-Z0/Slopebuoy so Lb = Lbb = Lx-LxmLb
        W0 = 0.0
        lambd = 10
        dxx = Lx/nx
        muu = 5*dxx
        alphaa = 5*dxx
        betaa = 5*dxx

    n_mode = Lx/lambd #
    print('n_mode',n_mode)
    kx = 2 * np.pi * n_mode / Lx
    omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
    
    Tperiod = 2.0*np.pi/omega
    tstop = 9*Tperiod
    tstop = 19*Tperiod
    tstop = Tperiod
    tstop = 28*Tperiod
    nTfac = 30
    nTfac = 19
    nTfac = 2
    tstop = Tperiod
    t_end = nTfac*Tperiod # time of simulation [s]
    Tstartmeas = 0*Tperiod
    # t_end = 0.1*Tperiod # waveflap test
    
    dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
    Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
    CFL = 0.125 # run at a) 0.125 and b) 0.5*0.125
    CFL = 1.0
    dt = CFL*Tperiod/Nt  # 0.005  # time step [s]
    print('dtt=',dtt, t_end/dtt,dt,2/omega)
    D = 0.0
    phi_exact_expr = D * x[0] * x[1]
    phi_exact_exprH0 = D * x[0]
    eta_exact_expr = D * x[0]
    
    if nvpcase == 24: # Hard constraint or Archimedes principle rest solution wave-buoy system
        eta_exact_expr = D * x[0]
        eta_exact_expr = eta_exact_expr + (Z0-Keel+Slopebuoy*(Lx-x[0])-H0)*0.5*(1.0+fd.sign(x[0]-Lbb)) # hb(Z,x)-H0 when x > Lbb
        
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+1*dt):
        time.append(t)
        t+= dt
    # nplot = 8*nTfac
    nplot = 4*nTfac
    # nplot = 4 # waveflap test
    nplot = nTfac
    nplot = 4*nTfac
    
    
dtmeet = t_end/nplot # (0:nplot)*dtmeet
tmeet = dtmeet
tmeas = np.linspace(0.0, t_end, nplot+1)
print(' S: tmeet gvd', dtmeet, tmeet)
print('tmeas', tmeas)
epsmeet = 10.0**(-10)
nt = int(len(time)/nplot)
t_plot = time[0::nt]
print('t_plot', t_plot,nt,nplot, t_end)
print('gg:',gg)
color = np.array(['g-', 'b--', 'r:', 'm:'])
colore = np.array(['k:', 'c--', 'm:'])


##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

fig, (ax1, ax2) = plt.subplots(2)

# 
if nvpcase == 111:
    ax1.set_title(r'VP linear SE steps 1+2, 3:',fontsize=tsize2)
elif nvpcase == 2:
    ax1.set_title(r'VP nonlinear SE steps 1+2, 3:',fontsize=tsize2)
elif nvpcase == 21:
    ax1.set_title(r'VP nonlinear SE steps 1+2, 3, wavemaker:',fontsize=tsize2)
elif nvpcase == 22:
    # ax1.set_title(r'VP nonlinear SV steps 1+2, 3, 4, wavemaker:',fontsize=tsize2)
    ax1.set_title(r'VP nonlinear SV, wavemaker:',fontsize=tsize2)
elif nvpcase == 23:
    # ax1.set_title(r'VP nonlinear midpoint, 1+2+3, wavemaker:',fontsize=tsize2)
    ax1.set_title(r'VP nonlinear midpoint, wavemaker:',fontsize=tsize2)
elif nvpcase == 233:
    # ax1.set_title(r'VP nonlinear midpoint, 1+2+3, wavemaker:',fontsize=tsize2)
    if nowaveflap==1:
        ax1.set_title(r'VP nonlinear midpoint, waveflap:',fontsize=tsize2)
    if nowaveflap==0:
        ax1.set_title(r'VP nonlinear midpoint, waveflap/piston:',fontsize=tsize2)
elif nvpcase == 24:
    ax1.set_title(r'VP nonlinear midpoint, 1+2+3, wavemaker/buoy:',fontsize=tsize2)
# End if
ax1.set_ylabel(r'$\eta (x,t) [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
ax2.grid()


#__________________  Define function spaces  __________________#

nCG = 2
nCGvert = 6
ntest = 0
if ntest==1: # not needed
    nCGvert = 2
    mini_horiz = fd.FiniteElement('CG', interval, nCG)
    mini_vert = fd.FiniteElement('CG', interval, nCGvert)
    mini_elt = fd.TensorProductElement(mini_horiz, mini_vert)
    V_W = fd.FunctionSpace(mesh, mini_elt)
elif ntest==0: # So with 1 element in vertical and NCGvert high-order one has Gidel's-code if f(z) is made properly.
    V_W = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='CG', vdegree=nCGvert) # interior potential varphi; can mix degrees in hor and vert
# End if
V_R = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='R', vdegree=0) # free surface eta and surface potential phi extended uniformly in vertical: vdegree=0
V_C = fd.FunctionSpace(mesh, 'R', 0, vfamily='R', vdegree=0) # buoy variables W and Z

varphi = fd.Function(V_W, name="varphi")

phi_f = fd.Function(V_R, name="phi_f")  # at the free surface
phiii = fd.Function(V_R, name="phi")
phif_new = fd.Function(V_R, name="phi_f") 

eta = fd.Function(V_R, name="eta")
eta_new = fd.Function(V_R, name="eta_new")

heta = fd.Function(V_R, name="heta")
h_new = fd.Function(V_R, name="h_new")

v_W = fd.TestFunction(V_W)
v_R = fd.TestFunction(V_R)

# Variables for Stormer-Verlet waves
mixed_V = V_R * V_W
# phi, varphi = fd.Function(mixed_V)
result_mixed = fd.Function(mixed_V)
vvp = fd.TestFunction(mixed_V)
vvp0, vvp1 = fd.split(vvp)  # These represent "blocks".
phii, varphii = fd.split(result_mixed)

# Variables for modified midpoint waves
mixed_Vmp = V_R * V_R * V_W
result_mixedmp = fd.Function(mixed_Vmp)
vvmp = fd.TestFunction(mixed_Vmp)
vvmp0, vvmp1, vvmp2 = fd.split(vvmp)  # These represent "blocks".
phimp, etamp, varphimp= fd.split(result_mixedmp)

# Variables for  wave-buoy case ONNO 16012023: if not in an if case 23 does not work anymore because phimp, etamp, varphimp are then defined otherwise in 24!
if nvpcase==24:
    Wznew = fd.Function(V_C, name="Wznew")
    Zznew = fd.Function(V_C, name="Zznew")
    Wz = fd.Function(V_C, name="Wz")
    Zz = fd.Function(V_C, name="Zz")
    mixed_Vmpb = V_C * V_C * V_R * V_R * V_W
    result_mixedmpb = fd.Function(mixed_Vmpb)
    Zzmp, Wzmp, phimp, etamp, varphimp= fd.split(result_mixedmpb)
    vvmpb = fd.TestFunction(mixed_Vmpb)
    vb0, vb1, vvmpb0, vvmpb1, vvmpb2 = fd.split(vvmpb)  # These represent "blocks", the specific test functions


# phi12 = fd.Function(V_R, name="phi12") # eta12 = fd.Function(V_R, name="eta12")


##_________________  Boundary Conditions __________________________##

class MyBC(fd.DirichletBC):
    def __init__(self, V, value, markers):
        super(MyBC, self).__init__(V, value, 0)
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])

def surface_BC():
    bc = fd.DirichletBC(V_R, 1, top_id)
    f = fd.Function(V_R, dtype=np.int32)
    ## f is now 0 everywhere, except on the boundary
    bc.apply(f)
    return MyBC(V_R, 0, f)

def surface_BC_mixed(): # ONNO 06-01 I do not understand what is inside: surface_BC() or MyBC() magic; no idea where explanation is; should this be 1 instead of 0; no!
        bc_mixed = fd.DirichletBC(mixed_V.sub(0), 1, top_id)
        f_mixed = fd.Function(mixed_V.sub(0), dtype=np.int32)
        bc_mixed.apply(f_mixed)
        return MyBC(mixed_V.sub(0), 0, f_mixed)

BC_exclude_beyond_surface = surface_BC()
BC_exclude_beyond_surface_mixed = surface_BC_mixed() 

#                                      
BC_phi_f = fd.DirichletBC(V_R, phi_f, top_id)
BC_phif_new = fd.DirichletBC(V_R, phif_new, top_id)
# BC_phi = fd.DirichletBC(V_W, phi, top_id)
BC_varphi = fd.DirichletBC(V_W, 0, top_id)
BC_varphi_mixed = fd.DirichletBC(mixed_V.sub(1), 0, top_id)  # Wave SV
# bc0 = DirichletBC(W.sub(0), as_vector([0.0, -sin(5*x)]), 3)

BC_varphi_mixedmp = fd.DirichletBC(mixed_Vmp.sub(2), 0, top_id) # Wave modified midpoint

if nvpcase==24:
    BC_varphi_mixedmpb = fd.DirichletBC(mixed_Vmpb.sub(4), 0, top_id) # Wave-buoy modified midpoint

# 
#
#
if nvpcase==111: # as 11 but steps 1 and 2 in combined solve; step 3 separately 
    VP11 = ( fd.inner(phii, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * gg * fd.inner(eta, eta)) )* fd.ds_t \
        - ( 1/2 * fd.inner(fd.grad(phii+varphii), fd.grad(phii+varphii))  ) * fd.dx
    
    # Step-1 and 2 must be solved in tandem: f-derivative VP wrt eta to find update of phi at free surface
    # int -phi/dt + phif/dt - gg*et) delta eta ds a=0 -> (phi-phif)/dt = -gg * eta
    phif_expr1 = fd.derivative(VP11, eta, du=vvp0)  # du represents perturbation

    # Step-2: f-derivative VP wrt varphi to get interior phi given surface update phi
    # int nabla (phi+varphi) cdot nabla delta varphi dx = 0
    phi_expr1 = fd.derivative(VP11, varphii, du=vvp1)
    Fexpr = phif_expr1+phi_expr1
    phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixed, bcs = BC_varphi_mixed))
    # [BC_exclude_beyond_surface_mixed,BC_varphi_mixed])) #  not needed don't set any exclude http://firedrakeproject.org/variational-problems.html#id22
    # BC_varphi_mixed sets it for 2nd variable varphi with no. 1
     
    # 
    # Step-3: f-derivative wrt phi but restrict to free surface to find updater eta_new; only solve for eta_new by using exclude
    eta_expr2 = fd.derivative(VP11, phii, du=v_R)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr2,eta_new)) #  ,bcs=BC_exclude_beyond_surface)) # not needed so omitted
if nvpcase==1112: # as 11 but steps 1 and 2 in combined solve; step 3 separately; wavemaker
    VP11 = ( fd.inner(phii, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * gg * fd.inner(eta, eta)) )* fd.ds_t \
        - ( 1/2 * fd.inner(fd.grad(phii+varphii), fd.grad(phii+varphii))  ) * fd.dx
    
    # Step-1 and 2 must be solved in tandem: f-derivative VP wrt eta to find update of phi at free surface
    # int -phi/dt + phif/dt - gg*et) delta eta ds a=0 -> (phi-phif)/dt = -gg * eta
    phif_expr1 = fd.derivative(VP11, eta, du=vvp0)  # du represents perturbation

    # Step-2: f-derivative VP wrt varphi to get interior phi given surface update phi
    # int nabla (phi+varphi) cdot nabla delta varphi dx = 0
    phi_expr1 = fd.derivative(VP11, varphii, du=vvp1)
    Fexpr = phif_expr1+phi_expr1
    phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixed, bcs = BC_varphi_mixed))
    # [BC_exclude_beyond_surface_mixed,BC_varphi_mixed])) #  not needed don't set any exclude http://firedrakeproject.org/variational-problems.html#id22
    # BC_varphi_mixed sets it for 2nd variable varphi with no. 1
     
    # 
    # Step-3: f-derivative wrt phi but restrict to free surface to find updater eta_new; only solve for eta_new by using exclude
    eta_expr2 = fd.derivative(VP11, phii, du=v_R)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr2,eta_new)) #  ,bcs=BC_exclude_beyond_surface)) # not needed so omitted
elif nvpcase==2: # Steps 1 and 2 need solving in unison
    # 
    # Desired VP format of the above
    param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    # 
    # VP formulation of above with phi^(n+1)=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    # 
    Lw = 0.5*Lx
    Ww = Lw # Later wavemaker to be added # eta_new -> h_new and eta -> heta ; Nonlinear potential-flow VP:
    facc = 1.0
    faccc = 1.0
    fac = 1.0 # now same as linear case above except for constant pref-factors as check; 
    VPnl = ( H0*Ww*fd.inner(phii, (eta_new - eta)/dt) + H0*Ww*fd.inner(phi_f, eta/dt) - gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) )* fd.ds_t \
        - 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                  + Ww * (H0**2/(H0+fac*eta)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx
    #  Step-1: only nonlinear step just trying these solver_parameters!    
    phif_exprnl1 = fd.derivative(VPnl, eta, du=vvp0) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?

    #  Step-2: linear solve; 
    phi_exprnl1 = fd.derivative(VPnl, varphii, du=vvp1)

    Fexprnl = phif_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixed, bcs = BC_varphi_mixed), solver_parameters=param_psi)

    #  Step-3: linear solve; 
    heta_exprnl2 = fd.derivative(VPnl, phii, du=v_R)
    heta_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(heta_exprnl2,eta_new))
    #
elif nvpcase==21: # SE Steps 1 and 2 need solving in unison case with wavemaker initial condition nic=1
    # Using ideas form here for time-dependence of wavemaker: https://www.firedrakeproject.org/demos/higher_order_mass_lumping.py.html
    # Desired VP format of the above
    param_psi1 = {'ksp_converged_reason':None,'ksp_type': 'preonly', 'pc_type': 'lu'}
    param_psi2  = {'ksp_converged_reason':None,'ksp_type': 'preonly', 'pc_type': 'lu','snes_type': 'newtonls'}
    param_psi  = {'ksp_type': 'preonly', 'pc_type': 'lu','snes_type': 'newtonls','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'}
    param_psi5 = {'ksp_converged_reason':None, 'snes_type': 'newtonls','ksp_type': 'gmres', 'pc_type': 'jacobi'}
    t = 0
    gam = 0.05
    sigm = omega
    gam = 0.5 #0.002 gam = 0.7
    
    tstop = 9*Tperiod
    tstop = Tperiod
    
    def Rwavemaker(t,gam,sigm,tstop):
        Rh1 = -gam*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = -gam*fd.cos(sigm*tstop)
        return Rh1
    def dRwavemakerdt(t,gam,sigm,tstop):
        Rt1 = gam*sigm*fd.sin(sigm*t)         
        if t >= tstop:
            Rt1 = 0.0*gam*sigm*fd.sin(sigm*tstop)
        return Rt1
    Rwave = fd.Constant(0.0)
    Rwave.assign(Rwavemaker(t,gam,sigm,tstop)) 
    dRwavedt = fd.Constant(0.0)
    dRwavedt.assign(dRwavemakerdt(t,gam,sigm,tstop))
    Lw = 0.5*Lx # Error? I did not put in the break so really should choose Lw=Lx
    Ww = fd.Constant(0.0)
    Wwn = fd.Constant(0.0)
    Ww.assign(Lw-Rwavemaker(t,gam,sigm,tstop))      #  Lw-Ww
    Wwn.assign(Lw-Rwavemaker(t-1.0*dt,gam,sigm,tstop))      #  Lw-Wwn

    # 
    # VP formulation of above with phi^(n+1)=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    # 
    # Ww = Lw - Rwave # With wavemaker at n+1/2 # eta_new -> h_new and eta -> heta ; how does one include time dependence in VP?
    # Nonlinear potential-flow VP:
    facc = 1.0
    faccc = 1.0
    fac = 1.0 # now same as linear case above except for constant pref-factors as check; 
    VPnl = ( H0*Ww*fd.inner(phii, (eta_new - eta)/dt) + H0*Wwn*fd.inner(phi_f, eta/dt) - gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) \
             -H0*phii*(x[0]-Lw)*dRwavedt*eta.dx(0) )* fd.ds_t \
        - 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                  + Ww * (H0**2/(H0+fac*eta)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx - Lw*dRwavedt*(phii+varphii)* (H0+eta) * fd.ds_v(1) # added (H0+eta) was missing 10-01
    #  Step-1: only nonlinear step just trying these solver_parameters!    
    phif_exprnl1 = fd.derivative(VPnl, eta, du=vvp0) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?

    #  Step-2: solve; 
    phi_exprnl1 = fd.derivative(VPnl, varphii, du=vvp1)

    Fexprnl = phif_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixed, bcs = BC_varphi_mixed), solver_parameters=param_psi)

    #  Step-3: linear solve; 
    heta_exprnl2 = fd.derivative(VPnl, phii, du=v_R)
    heta_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(heta_exprnl2,eta_new))
    #
elif nvpcase==22: # Steps 1 and 2 need solving in unison; Stormer-Verlet
    # 
    # Desired VP format of the above
    param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    # 
    # VP formulation of above with phi^(n+1)=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    #
    Lw = 0.5*Lx # Error? I did not put in the break so really should choose Lw=Lx
    Lw = Lx
    t = 0
    gam = 0.05
    sigm = omega
    gam = 0.5 #0.002 gam = 0.7
    
    tstop = 9*Tperiod
    tstop = Tperiod
    
    def Rwavemaker(t,gam,sigm,tstop):
        Rh1 = -gam*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = -gam*fd.cos(sigm*tstop)
        return Rh1
    def dRwavemakerdt(t,gam,sigm,tstop):
        Rt1 = gam*sigm*fd.sin(sigm*t)         
        if t >= tstop:
            Rt1 = 0.0*gam*sigm*fd.sin(sigm*tstop)
        return Rt1
    Rwave = fd.Constant(0.0)
    Rwave.assign(Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
    dRwavedt = fd.Constant(0.0)
    dRwavedt.assign(dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
    Lw = 0.5*Lx # Error? I did not put in the break so really should choose Lw=Lx
    Lw = Lx
    Ww = fd.Constant(0.0) # Error? I did not put in the break so really should choose Lw=Lx
    Wwn = fd.Constant(0.0) # Error? I did not put in the break so really should choose Lw=Lx
    Wwp = fd.Constant(0.0) # Error? I did not put in the break so really should choose Lw=Lx
    Ww.assign(Lw-Rwavemaker(t+0.5*dt,gam,sigm,tstop))      #  Lw-Ww n+1/2
    Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))     #  Lw-Wwn n
    Wwp.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))     #  Lw-Wwn n+1
    
    # Ww = Lw  Later wavemaker to be added # eta_new -> h_new and eta -> heta ; Nonlinear potential-flow VP:
    facc = 1.0
    faccc = 1.0
    fac = 1.0 # now same as linear case above except for constant pref-factors as check;
    #
    # phii = psi^n+1/2; phi_f = psi^n; phiii= psi^n+1
    #
    VPnl = ( H0*Ww*fd.inner(phii, (eta_new - eta)/dt) -H0*Wwp*fd.inner(phiii,eta_new/dt) + H0*Wwn*fd.inner(phi_f, eta/dt) \
             - 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta_new, H0+eta_new)-(H0+eta_new)*H0+0.5*H0**2) \
             - 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) \
             + 0.5*H0*phii*(x[0]-Lw)*dRwavedt*(eta.dx(0)+eta_new.dx(0)) )* fd.ds_t \
             - 0.25 * ( (Lw**2/Ww) * (H0+fac*eta_new) * (phii.dx(0)+varphii.dx(0)-(x[1]/(H0+fac*eta_new))*fac*eta_new.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                       +(Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(x[1]/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)) )**2 \
                       + Ww*(H0**2/(H0+fac*eta)+H0**2/(H0+fac*eta_new)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx - 0.5*Lw*dRwavedt*(phii+varphii)*(H0+eta+H0+eta_new)*fd.ds_v(1)
    #  Step-1: only nonlinear step just trying these solver_parameters!    
    phif_exprnl1 = fd.derivative(VPnl, eta, du=vvp0) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?

    #  Step-2: solve; for phi^{n+1/2} and varphi^{n+1/2}
    phi_exprnl1 = fd.derivative(VPnl, varphii, du=vvp1)

    Fexprnl = phif_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixed, bcs = BC_varphi_mixed), solver_parameters=param_psi)

    #  Step-3: solve; for h^{n+1}
    heta_exprnl2 = fd.derivative(VPnl, phii, du=v_R)
    heta_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(heta_exprnl2,eta_new))

    #  Step-4: linear solve; for phi^{n+1}
    phif_exprnl4 = fd.derivative(VPnl, eta_new, du=v_R) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?
    phif_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phif_exprnl4 ,phiii))
elif nvpcase==233: # Steps 1, 2, 3 solved in unison; implicit mid point case
    # 
    # Desired VP format of the above
    param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    # 
    # VP formulation of above with phi^(n+1)=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    #
    Lw = Lx # Lw = 0.5*Lx # Error? I did not put in the break so really should choose Lw=Lx
    t = 0.0
    gam = 0.05
    sigm = omega
    gam = 0.1 #0.002 gam = 0.7
    gam = 0.5 #
    gam = 1.0 # waveflap test
    gam = 0.2 # waveflap test gam = 1. # waveflap test
    
    def Rwavemaker(t,gam,sigm,tstop):
        Rh1 = -gam*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = -gam*fd.cos(sigm*tstop)
        return Rh1
    def dRwavemakerdt(t,gam,sigm,tstop):
        Rt1 = gam*sigm*fd.sin(sigm*t)         
        if t >= tstop:
            Rt1 = 0.0*gam*sigm*fd.sin(sigm*tstop)
        return Rt1
    def Wflap(t,gam,sigm,tstop): # W(z,t) = z*gam*sin(sigm*t) = z*Wflap(t)
        # Piston case as first test
        Rh1 = -gam*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = -gam*fd.cos(sigm*tstop)
        return Rh1
    def dWflapdt(t,gam,sigm,tstop):
        # Piston case as first test
        Rt1 = gam*sigm*fd.sin(sigm*t)
        if t >= tstop:
            Rt1 = 0.0
        return Rt1
    def dWflapdz(t,gam,sigm,tstop):
        # Piston case as first test
        fac = 0.0
        Rt1 = -fac*gam*fd.cos(sigm*t)
        if t >= tstop:
            Rt1 = -fac*gam*fd.cos(sigm*tstop)
        return Rt1
    
    Rwave = fd.Constant(0.0)
    Rwave.assign((1-nowaveflap)*Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
    dRwavedt = fd.Constant(0.0)
    dRwavedt.assign((1-nowaveflap)*dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
    Lw = Lx # Error? I did not put in the break so really should choose Lw=Lx
    Wflaps = fd.Constant(0.0)
    dWflapdts = fd.Constant(0.0)
    dWflapdzs = fd.Constant(0.0)
    Wflaps.assign(nowaveflap*Wflap(t+0.5*dt,gam,sigm,tstop))
    dWflapdts.assign(nowaveflap*dWflapdt(t+0.5*dt,gam,sigm,tstop))
    dWflapdzs.assign(nowaveflap*dWflapdz(t+0.5*dt,gam,sigm,tstop))
    Fz = x[1]*(H0+etamp)/H0
    FzH0 = (H0+etamp)
    FtnWM = Fz # z-dependence of waveflap wavemaker
    dFtnWMdz = 1.0 # dz-dependence of waveflap wavemaker
    FtnWMH0 = FzH0 # z-dependence of waveflap wavemaker at free surface H0
    dFtnWMH0dz = 1.0 # dz-dependence of waveflap wavemaker at free surface H0
    Fdxdxi = (1.0-( Rwave+FtnWM*Wflaps )/Lw) + (x[1]/H0)*( dWflapdzs )*(1-x[0]/Lw)*etamp.dx(0)
    Fdxdxi3 = (1.0-x[0]/Lw)*( dWflapdzs )*(H0+etamp)/H0
    Fdzdxi = x[1]*etamp.dx(0)/H0
    Fdzdxi3 = (H0+etamp)/H0
    FJacobian = (Fdxdxi*Fdzdxi3 - Fdxdxi3*Fdzdxi)
    Fpsi = phimp/(1.0-( Rwave+FtnWM*Wflaps )/Lw) # wrong old not used
    Fpsi0 = phimp/(1.0-( Rwave+FtnWMH0*Wflaps )/Lw) # at free surface
    Fdpsidxi = (phimp.dx(0)/(1.0-( Rwave+FtnWM*Wflaps )/Lw)+phimp/((1.0-( Rwave+FtnWM*Wflaps )/Lw)**2))*etamp.dx(0)*x[1]*( dWflapdzs )/(Lw*H0) # old
    Fdpsidxi0 = phimp.dx(0)/(1.0-( Rwave+FtnWMH0*Wflaps )/Lw)+( phimp/((1.0-( Rwave+FtnWMH0*Wflaps )/Lw)**2) )*etamp.dx(0)*( dWflapdzs )/Lw
    Fdpsidxi3 = 0.0  # (phimp.dx(1)/(1.0-( Rwave+FtnWM*Wflaps )/Lw)+(phimp/((1.0-( Rwave+FtnWM*Wflaps )/Lw)**2))*(H0+etamp)*( dWflapdzs )/Lw*H0)
    # KOKI2102
    if nz==1:
        phihat = fd.product( (x[1]-(H0/nCGvert)*(nCGvert+1-kk))/(H0-(H0/nCGvert)*(nCGvert+1-kk)) for kk in range(2,nCGvert+1,1) ) # function phihat
        # zz = sp.Symbol('zz')
        #  phihatt = sp.prod( (zz-(H0/nCGvert)*(nCGvert+1-kk))/(H0-(H0/nCGvert)*(nCGvert+1-kk)) for kk in range(2,nCGvert+1,1) ) # function phihat
        #  phihat = fd.replace(phihatt,{zz:x[1]})
        #  dphihatdx3 = sp.diff(phihatt,zz) # derivative wrt x[1] of function phihat
        #  dphihatdxi3 = replace(dphihatdx3,{zz:x[1]}) # derivative wrt x[1] of function phihat
        # dphihatdxi3 = phihat*sum( 1.0/(x[1]-(H0/nCGvert)*(nCGvert+1-kk)) for kk in range(2,nCGvert+1,1) ) #
        dphihatdxi3 = fd.derivative(phihat, x[1], du=v_W)
    else:
        phihat = 1.0 # function phihat
        dphihatdxi3 = 0.0 # + 0.0*X[1] # KOKI2102 sp.diff(phihat,x[1]) # derivative wrt x[1] of function phihat
    #         
    # Midpoint:
    # phimp = psi^n+1/2; phi_f = psi^n; eta = eta^n; etamp= psi^n+1/2; eta = eta^(n+1}; phiii = phi^{n+1}; varphimp = varphi^(n+1/2} aid variable
    #
    VPnl = ( -fd.inner(phimp, (eta_new - eta)/dt) + fd.inner(etamp,(phiii-phi_f)/dt) \
             + gg*Fdxdxi*( 0.5*fd.inner(Fz,Fz)-Fz*H0+0.5*H0**2 ) \
             + Fpsi0*(1.0-x[0]/Lw)*( dRwavedt+FtnWM*dWflapdts )*etamp.dx(0) )  * fd.ds_t \
             + 0.5 * (1.0/FJacobian)* ( (Fdxdxi3**2+Fdzdxi3**2) * ( Fdpsidxi0*phihat+varphimp.dx(0) )**2  \
                                    -2.0 * (Fdxdxi*Fdxdxi3+Fdzdxi*Fdzdxi3) * ( Fdpsidxi0*phihat+varphimp.dx(0) ) * ( Fdpsidxi3*dphihatdxi3+varphimp.dx(1) ) \
                                    + (Fdxdxi**2+Fdzdxi**2)*( Fdpsidxi3*dphihatdxi3+varphimp.dx(1) )**2 ) * fd.dx \
                                    + ( Fdzdxi3*( dRwavedt+FtnWM*dWflapdts ) *( Fpsi0*phihat+varphimp ) + Fdxdxi3*gg*(0.5*Fz**2-H0*Fz) ) * fd.ds_v(1)
    
    #  Step-1: only nonlinear step just trying these solver_parameters! Wrt phimp = phi^(n+1/2} for eta^(n+1) = 2*eta^(n+1/2)-eta^n
    phif_exprnl1 = fd.derivative(VPnl, phimp, du=vvmp0) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?
    phif_exprnl1 = fd.replace(phif_exprnl1, {phiii: 2.0*phimp-phi_f})
    phif_exprnl1 = fd.replace(phif_exprnl1, {eta_new: 2.0*etamp-eta}) 

    #  Step-2: solve; for phi^{n+1/2} and varphi^{n+1/2} Wrt varmp=varphi^(n+1/2) for varmp=varphi^(n+1/2
    phi_exprnl1 = fd.derivative(VPnl, varphimp, du=vvmp2)
    phi_exprnl1 = fd.replace(phi_exprnl1, {phiii: 2.0*phimp-phi_f})
    phi_exprnl1 = fd.replace(phi_exprnl1, {eta_new: 2.0*etamp-eta}) 

    #  Step-3: solve; for phi^{n+1/2} and varphi^{n+1/2} Wrt etamp=eta^(n+1/2) for phiii=phi^(n+1)=2*phi^(n+1/2)-phi^b
    eta_exprnl1 = fd.derivative(VPnl, etamp, du=vvmp1)
    eta_exprnl1 = fd.replace(eta_exprnl1, {phiii: 2.0*phimp-phi_f})
    eta_exprnl1 = fd.replace(eta_exprnl1, {eta_new: 2.0*etamp-eta}) 

    Fexprnl = phif_exprnl1+phi_exprnl1+eta_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=param_psi)

elif nvpcase==23: # Steps 1 and 2 need solving in unison; mid point ONO 22-01-2023: Working case for storage
    # 
    # Desired VP format of the above
    param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    # 
    # VP formulation of above with phi^(n+1)=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    #
    Lw = 0.5*Lx # Error? I did not put in the break so really should choose Lw=Lx
    Lw = Lx
    t = 0
    gam = 0.05
    sigm = omega
    gam = 0.1 #0.002 gam = 0.7
    gam = 0.5
    
    def Rwavemaker(t,gam,sigm,tstop):
        Rh1 = -gam*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = -gam*fd.cos(sigm*tstop)
        return Rh1
    def dRwavemakerdt(t,gam,sigm,tstop):
        Rt1 = gam*sigm*fd.sin(sigm*t)         
        if t >= tstop:
            Rt1 = 0.0*gam*sigm*fd.sin(sigm*tstop)
        return Rt1
    
    Rwave = fd.Constant(0.0)
    Rwave.assign(Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
    dRwavedt = fd.Constant(0.0)
    dRwavedt.assign(dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
    Lw = 0.5*Lx
    Lw = Lx
    Ww = fd.Constant(0.0)
    Wwn = fd.Constant(0.0)
    Wwp = fd.Constant(0.0)
    Ww.assign(Lw-Rwavemaker(t+0.5*dt,gam,sigm,tstop)) #  Lw-Ww n+1/2
    Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop)) #  Lw-Wwn n
    Wwp.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop)) #  Lw-Wwn n+1
    
    # Ww = Lw  Later wavemaker to be added # eta_new -> h_new and eta -> heta ; Nonlinear potential-flow VP:
    facc = 1.0
    faccc = 1.0
    fac = 1.0 # now same as linear case above except for constant pref-factors as check;
    # Midpoint:
    # phimp = psi^n+1/2; phi_f = psi^n; eta = eta^n; etamp= psi^n+1/2; eta = eta^(n+1}; phiii = phi^{n+1}; varphimp = varphi^(n+1/2} aid variable
    # # not needed: - H0*Wwp*fd.inner(phiii,(eta_new-eta)/dt) + H0*Wwn*fd.inner(phi_f, eta/dt) \
    VPnl = ( H0*Ww*fd.inner(phimp, (eta_new - eta)/dt) \
             - H0*fd.inner(etamp,(Wwp*phiii-Wwn*phi_f)/dt) \
             - gg*Ww*H0*( 0.5*fd.inner(H0+etamp, H0+etamp)-(H0+etamp)*H0+0.5*H0**2 ) \
             + fac*H0*phimp*(x[0]-Lw)*dRwavedt*etamp.dx(0) )  * fd.ds_t \
             - 0.5 * ( (Lw**2/Ww) * (H0+fac*etamp) * (phimp.dx(0)+varphimp.dx(0)-(z/(H0+fac*etamp))*fac*etamp.dx(0)*(facc*phimp.dx(1)+varphimp.dx(1)))**2 \
                       + Ww*(H0**2/(H0+fac*etamp)) * (faccc*phimp.dx(1)+varphimp.dx(1))**2 ) * fd.dx \
                       - Lw*dRwavedt*(phimp+varphimp)* (H0+fac*etamp)*fd.ds_v(1)
    #  Step-1: only nonlinear step just trying these solver_parameters! Wrt phimp = phi^(n+1/2} for eta^(n+1) = 2*eta^(n+1/2)-eta^n
    phif_exprnl1 = fd.derivative(VPnl, phimp, du=vvmp0) # du=v_W represents perturbation seems that with these solver_parameters solves quicker: tic-toc it with and without?
    phif_exprnl1 = fd.replace(phif_exprnl1, {phiii: 2.0*phimp-phi_f})
    phif_exprnl1 = fd.replace(phif_exprnl1, {eta_new: 2.0*etamp-eta}) 

    #  Step-2: solve; for phi^{n+1/2} and varphi^{n+1/2} Wrt varmp=varphi^(n+1/2) for varmp=varphi^(n+1/2
    phi_exprnl1 = fd.derivative(VPnl, varphimp, du=vvmp2)
    phi_exprnl1 = fd.replace(phi_exprnl1, {phiii: 2.0*phimp-phi_f})
    phi_exprnl1 = fd.replace(phi_exprnl1, {eta_new: 2.0*etamp-eta}) 

    #  Step-3: solve; for phi^{n+1/2} and varphi^{n+1/2} Wrt etamp=eta^(n+1/2) for phiii=phi^(n+1)=2*phi^(n+1/2)-phi^b
    eta_exprnl1 = fd.derivative(VPnl, etamp, du=vvmp1)
    eta_exprnl1 = fd.replace(eta_exprnl1, {phiii: 2.0*phimp-phi_f})
    eta_exprnl1 = fd.replace(eta_exprnl1, {eta_new: 2.0*etamp-eta}) 

    Fexprnl = phif_exprnl1+phi_exprnl1+eta_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=param_psi)
# End if
#  
# tmp_eta, tmp_phi = result_mixed.split()

phii, varphii = result_mixed.split()

bc_top = fd.DirichletBC(V_R, 0, top_id)
#  eta.assign(0.0)
#  phi.assign(0.0)
eta_exact = fd.Function(V_R)
eta_exact.interpolate(eta_exact_expr)
eta.interpolate(eta_exact)
#  phi.interpolate(phi_exact_expr)
phi_f.interpolate(phi_exact_expr)
phii.interpolate(phi_exact_exprH0)
varphii.interpolate(phi_exact_expr-phi_exact_exprH0)
if nvpcase==22:
    eta_new.interpolate(eta_exact)
elif nvpcase == 24:
    Zz = Z0 # .interpolate(Z0)
    Wz = W0 # .interpolate(W0)
elif nvpcase==233:
    eta_exact.interpolate(eta_exact_expr)
    eta.interpolate(eta_exact)
    Rwave.assign((1-nowaveflap)*Rwavemaker(0.0,gam,sigm,tstop))
    dRwavedt.assign((1-nowaveflap)*dRwavemakerdt(0.0,gam,sigm,tstop))
    Wflaps.assign(nowaveflap*Wflap(0.0,gam,sigm,tstop))
    Ffz = x[1]*eta/H0
    FfWflaps = nowaveflap*Wflaps 
    phi_f.interpolate(phi_exact_expr*(1.0-( Rwave+Ffz*Wflaps )/Lw)) # Projection on main variable at initial time
    phii.interpolate(phi_exact_exprH0) # For plotting only
# End if

###### OUTPUT FILES ##########
outfile_phi = fd.File("results/phi.pvd")
outfile_eta = fd.File("results/eta.pvd")
outfile_varphi = fd.File("results/varphi.pvd")

t = 0.0
i = 0.0


print('Plotting starts, initial data:')
plt.figure(1)
eta1vals = np.array([eta.at(x, zslice) for x in xvals]) #  pphi1vals = np.array([phii.at(xvals, zslice)])
phi1vals = np.array([phii.at(x, zslice) for x in xvals])

ax1.plot(xvals, eta1vals, ':k', label = f' $\eta_n: t = {t:.3f}$',linewidth=2)
ax2.plot(xvals, phi1vals, ':k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)

# output_data()
if nvpcase == 111:
    EKin = fd.assemble( 0.5*fd.inner(fd.grad(phii+varphii),fd.grad(phii+varphii))*fd.dx )
    EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds_t )
elif nvpcase == 2:    
    EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phii.dx(1))**2 + Ww * (H0**2/(H0+fac*eta)) * (phii.dx(1))**2) * fd.dx )
    EPot = fd.assemble( gg*Ww*H0*( 0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)
elif nvpcase == 21: 
    EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phii.dx(1))**2 + Ww * (H0**2/(H0+fac*eta)) * (phii.dx(1))**2) * fd.dx )
    EPot = fd.assemble( gg*Ww*H0*( 0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)
    # ?Kamil =  (gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) )* fd.ds_t \
    # ? + 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
    # ?        - Ww * (H0**2/(H0+fac*eta)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx + Lw*dRwavedt*(phii+varphii)* fd.ds_v(1)
    # ?Kamil = Kamil/(Lw*H0)
elif nvpcase == 22: 
    EKin = fd.assemble( 0.25 * ( (Lw**2/Ww) * (H0+fac*eta_new) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta_new))*fac*eta_new.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                 +(Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                 + Ww*(H0**2/(H0+fac*eta)+H0**2/(H0+fac*eta_new)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx )
    EPot = fd.assemble( (0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta_new, H0+eta_new)-(H0+eta_new)*H0+0.5*H0**2) \
                        + 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) ) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)
elif nvpcase == 23: 
    EKin = fd.assemble( 0.5 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phii.dx(1))**2 + Ww * (H0**2/(H0+fac*eta)) * (phii.dx(1))**2) * fd.dx )
    EPot = fd.assemble( gg*Ww*H0*( 0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)
elif nvpcase == 2333: 
    EKin = fd.assemble( 0.5 * ( (1.0/Ww) * (H0+fac*eta) * (phii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phii.dx(1))**2 + Ww * (H0/(H0+fac*eta)) * (phii.dx(1))**2) * fd.dx )
    EPot = fd.assemble( gg*Ww*( 0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
    EKin = EKin
    EPot = EPot
elif nvpcase == 233: 
    Rwave.assign((1-nowaveflap)*Rwavemaker(0.0,gam,sigm,tstop))
    Wflaps.assign(nowaveflap*Wflap(0.0,gam,sigm,tstop))
    dWflapdzs.assign(nowaveflap*dWflapdz(0.0,gam,sigm,tstop))
    Ffz = x[1]*(H0+eta)/H0
    FfzH0 = (H0+eta)
    FftnWM = Ffz #` z-dependence of waveflap wavemaker
    dFftnWMdz = 1.0 # dz-dependence of waveflap wavemaker
    Ffdxdxi = (1.0-( Rwave+FftnWM*Wflaps )/Lw)+(x[1]/H0)*( dWflapdzs )*(1-x[0]/Lw)*eta.dx(0)
    Ffdxdxi3 = (1.0-x[0]/Lw)* ( dWflapdzs )*(H0+eta)/H0
    Ffdzdxi = x[1]*eta.dx(0)/H0
    Ffdzdxi3 = (H0+eta)/H0
    FfJacobian = Ffdxdxi*Ffdzdxi3 - Ffdxdxi3*Ffdzdxi
    EKin = fd.assemble( 0.5 * (1.0/FfJacobian) * ( (Ffdxdxi3**2+Ffdzdxi3**2)*(phii.dx(0))**2 \
                                                  -2.0*(Ffdxdxi*Ffdxdxi3+Ffdzdxi*Ffdzdxi3)*(phii.dx(0))*(phii.dx(1)) \
                                                  + (Ffdxdxi**2+Ffdzdxi**2)*(phii.dx(1))**2) * fd.dx ) # Note ONNO 21-02-2023: incorrect if nonzero phi
    EPot = fd.assemble( gg*Ffdxdxi*( 0.5*fd.inner(Ffz,Ffz)-H0*Ffz+0.5*H0**2) * fd.ds_t )
    EKin = EKin
    EPot = EPot
elif nvpcase == 24:
    EKin = fd.assemble( 0.5 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phii.dx(1))**2 + Ww * (H0**2/(H0+fac*eta)) * (phii.dx(1))**2) * fd.dx )
    EPot = fd.assemble( (gg*Ww*H0*(0.5*fd.inner(H0+eta_new, H0+eta_new)-(H0+eta_new)*H0+0.5*H0**2) ) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)
    Epb = mm*gg*Zz # fd.assemble( mm*gg*Zz )
# End if

E = EKin+EPot
plt.figure(2)
if nvpcase == 24:
    Ebuoy = Ekb+Epb
    Ewave = E
    E = Ewave+Ebuoy
    plt.plot(t,Ekb,'.g')
    plt.plot(t,Epb,'.c')
    plt.plot(t,Ebuoy,'.k')
    plt.plot(t,Ewave,'.k')
    plt.plot(t,E,'.k')
    plt.plot(t,EPot,'.b')
    plt.plot(t,EKin,'.r')
    plt.ylabel(f'$E(t), K(t) (r), P(t) (b), e(t), k(t) (g), p(t) (c)$',fontsize=size)
else:
    E0 = E
    # plt.plot(t,E-E0,'.')
    plt.plot(t,E,'.k')
    plt.plot(t,EPot,'.b')
    plt.plot(t,EKin,'.r')
    plt.ylabel(f'$E(t), K(t) (r), P(t) (b)$',fontsize=size)
# End if
plt.xlabel(f'$t$ [s]',fontsize=size)
        
if nvpcase == 111:
    plt.title(r'Functional derivative VP used steps 1+2 & 3:',fontsize=tsize) # phi_expr.solve() # ?
elif nvpcase == 2:
    plt.title(r'VP nonlinear used steps 1+2 & 3:',fontsize=tsize)
elif nvpcase == 21: # plt.title(r'VP nonlinear used steps 1+2 & 3 wavemaker:',fontsize=tsize)
    plt.title(r'VP nonlinear SE, wavemaker:',fontsize=tsize)
    # ?figure(3)
    # ?plt.plot(t,Kamil,'.k')
elif nvpcase == 22: # plt.title(r'VP nonlinear used steps 1+2 & 3 wavemaker:',fontsize=tsize)
    plt.title(r'VP nonlinear SV, wavemaker:',fontsize=tsize)
elif nvpcase == 23:
    plt.title(r'VP nonlinear midpoint, wavemaker:',fontsize=tsize)
elif nvpcase == 233:
    if nowaveflap==1:
        plt.title(r'VP nonlinear midpoint, waveflap:',fontsize=tsize)
    elif nowaveflap==0:
        plt.title(r'VP nonlinear midpoint, waveflap/piston:',fontsize=tsize) 
# End if
print('E0=',E,EKin,EPot)

print('Time Loop starts')
# tic = time.time()
while t <= t_end + dt: #
    # print("time = ", t * T)
    tt = format(t, '.3f') 

    if nvpcase == 111: # VP linear steps 1 and 2 combined # solve of phi everywhere steps 1 and 2 combined
        phi_combo.solve() # 
        phii, varphii = result_mixed.split()
        eta_expr.solve()
    elif nvpcase == 2: # VP nonlinear steps 1 and 2 combined
        # phif_exprnl.solve() # solves phi^(n+1) at top free surface same as above
        # phi_exprnl.solve() # solves phi^(n+1) in interior and eta^(n+1) at top surface simulataneously
        phi_combonl.solve()
        phii, varphii = result_mixed.split()
        heta_exprnl.solve()
    elif nvpcase == 21: # VP nonlinear steps 1 and 2 combined with wavemaker
        Rwave.assign(Rwavemaker(t+1.0*dt,gam,sigm,tstop)) 
        dRwavedt.assign(dRwavemakerdt(t+1.0*dt,gam,sigm,tstop))
        Ww.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))      # Lw-Ww
        Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))      # Lw-Wwn
        phi_combonl.solve()
        phii, varphii = result_mixed.split()
        heta_exprnl.solve()
    elif nvpcase == 22:
        Rwave.assign(Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
        dRwavedt.assign(dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
        Ww.assign(Lw-Rwavemaker(t+0.5*dt,gam,sigm,tstop))      #  Lw-Ww n+1/2
        Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))     #  Lw-Wwn n
        Wwp.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))     #  Lw-Wwn n+1
        phi_combonl.solve()
        phii, varphii = result_mixed.split()
        heta_exprnl.solve()
        phif_exprnl.solve()
    elif nvpcase == 23:
        Rwave.assign(Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
        dRwavedt.assign(dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
        Ww.assign(Lw-Rwavemaker(t+0.5*dt,gam,sigm,tstop))      #  Lw-Ww n+1/2 used in weak forms 
        Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))     #  Lw-Wwn n used in weak forms 
        Wwp.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))     #  Lw-Wwn n+1 used in weak forms 
        phi_combonl.solve()
        phimp, etatmp, varphimp = result_mixedmp.split()
    elif nvpcase == 233:
        Rwave.assign((1-nowaveflap)*Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
        dRwavedt.assign((1-nowaveflap)*dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
        Wflaps.assign(nowaveflap*Wflap(t+0.5*dt,gam,sigm,tstop))
        dWflapdts.assign(nowaveflap*dWflapdt(t+0.5*dt,gam,sigm,tstop))
        dWflapdzs.assign(nowaveflap*dWflapdz(t+0.5*dt,gam,sigm,tstop))
        phi_combonl.solve()
        phimp, etatmp, varphimp = result_mixedmp.split()
    elif nvpcase == 24:
        Rwave.assign(Rwavemaker(t+0.5*dt,gam,sigm,tstop)) 
        dRwavedt.assign(dRwavemakerdt(t+0.5*dt,gam,sigm,tstop))
        Ww.assign(Lw-Rwavemaker(t+0.5*dt,gam,sigm,tstop))      #  Lw-Ww n+1/2
        Wwn.assign(Lw-Rwavemaker(t,gam,sigm,tstop))     #  Lw-Wwn n
        Wwp.assign(Lw-Rwavemaker(t+1.0*dt,gam,sigm,tstop))     #  Lw-Wwn n+1
        phi_combonl.solve()
        Zzmp, Wzmp, phimp, etatmp, varphimp = result_mixedmpb.split()
    # End if

    if nvpcase == 111:  # VP linear steps 1 and 2 combined
        phi_f.assign(phii)
        # phi.assign(phii+varphii)
        eta.assign(eta_new)
    elif nvpcase == 2: # VP nonlinear steps 1 and 2 combined
        phi_f.assign(phii)
        # phi.assign(phii+varphii)
        eta.assign(eta_new)
    elif nvpcase == 21: # VP nonlinear steps 1 and 2 combined
        phi_f.assign(phii)
        # phi.assign(phii+varphii)
        eta.assign(eta_new)
    # End if
    
    # Energy monitoring:
    if nvpcase == 111: # VP linear steps 1 and 2 combined
        EKin = fd.assemble( 0.5*fd.inner(fd.grad(phii+varphii),fd.grad(phii+varphii))*fd.dx )
        EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds_t )
    elif nvpcase == 2: # VP nonlinear steps 1 and 2 combined
        EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 + Ww * (H0**2/(H0+fac*eta)) * (facc*phii.dx(1)+varphii.dx(1))**2) * fd.dx )
        EPot = fd.assemble( gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
    elif nvpcase == 21: # VP nonlinear steps 1 and 2 combined
        EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 + Ww * (H0**2/(H0+fac*eta)) * (facc*phii.dx(1)+varphii.dx(1))**2) * fd.dx )
        EPot = fd.assemble( gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
    elif nvpcase == 22:
        EKin = fd.assemble( 0.25 * ( (Lw**2/Ww) * (H0+fac*eta_new) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta_new))*fac*eta_new.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                     +(Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 \
                                     + Ww*(H0**2/(H0+fac*eta)+H0**2/(H0+fac*eta_new)) * (faccc*phii.dx(1)+varphii.dx(1))**2 ) * fd.dx )
        EPot = fd.assemble( (0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta_new, H0+eta_new)-(H0+eta_new)*H0+0.5*H0**2) \
                            + 0.5*gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-(H0+eta)*H0+0.5*H0**2) )* fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
    elif nvpcase == 23:
        EKin = fd.assemble( 0.5 * ( (Lw**2/Ww) * (H0+fac*etamp) * (phimp.dx(0)+varphimp.dx(0)-(z/(H0+fac*etamp))*fac*etamp.dx(0)*(facc*phimp.dx(1)+varphimp.dx(1)))**2 \
                                     + Ww*(H0**2/(H0+fac*etamp)) * (faccc*phimp.dx(1)+varphimp.dx(1))**2 ) * fd.dx )
        EPot = fd.assemble( (gg*Ww*H0*(0.5*fd.inner(H0+etamp, H0+etamp)-(H0+etamp)*H0+0.5*H0**2) ) * fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
    elif nvpcase == 2333:
        EKin = fd.assemble( 0.5 * ( (1.0/Ww) * (H0+fac*etamp) * (phimp.dx(0)+varphimp.dx(0)-(z/(H0+fac*etamp))*fac*etamp.dx(0)*(facc*phimp.dx(1)+varphimp.dx(1)))**2 \
                                     + Ww*(H0/(H0+fac*etamp)) * (faccc*phimp.dx(1)+varphimp.dx(1))**2 ) * fd.dx )
        EPot = fd.assemble( (gg*Ww*(0.5*fd.inner(H0+etamp, H0+etamp)-(H0+etamp)*H0+0.5*H0**2) ) * fd.ds_t )
        EKin = EKin
        EPot = EPot
    elif nvpcase == 233:
        Rwave.assign((1-nowaveflap)*Rwavemaker(t+0.5*dt,gam,sigm,tstop))
        Wflaps.assign(nowaveflap*Wflap(t+0.5*dt,gam,sigm,tstop))
        dWflapdzs.assign(nowaveflap*dWflapdz(t+0.5*dt,gam,sigm,tstop))
        Ffz = x[1]*(H0+etamp)/H0
        FfzH0 = (H0+etamp)
        FftnWMH0 = FfzH0
        dFftnWMH0dz = 1.0
        Ffdxdxi = (1.0-( Rwave+Ffz*Wflaps )/Lw)+(x[1]/H0)*( dWflapdzs )*(1-x[0]/Lw)*etamp.dx(1)
        Ffdxdxi3 = (1.0-x[0]/Lw)*( dWflapdzs )*(H0+etamp)/H0
        Ffdzdxi = x[1]*etamp.dx(0)/H0
        Ffdzdxi3 = (H0+etamp)/H0
        FfJacobian = Ffdxdxi*Ffdzdxi3 - Ffdxdxi3*Ffdzdxi
        Ffpsi = phimp/(1.0-( Rwave+Ffz*Wflaps )/Lw)
        Ffpsi0 = phimp/(1.0-( Rwave+FfzH0*Wflaps )/Lw)
        Ffdpsidxi =  phimp.dx(0)/(1.0-( Rwave+Ffz*Wflaps )/Lw)+phimp/((1.0-( Rwave+Ffz*Wflaps )/Lw)**2)*etamp.dx(0)*x[1]*( dWflapdzs )/Lw
        Ffdpsidxi0 =  phimp.dx(0)/(1.0-( Rwave+Ffz*Wflaps )/Lw)+phimp/((1.0-( Rwave+Ffz*Wflaps )/Lw)**2)*etamp.dx(0)*x[1]*( dWflapdzs )/Lw
        Ffdpsidxi3 =  0.0  # phimp.dx(1)/(1.0-( Rwave+Ffz*Wflaps )/Lw)+phimp/((1.0-( Rwave+Ffz*Wflaps )/Lw)**2)*(H0+etamp)*( dWflapdzs )/Lw
        phihat = 1.0
        dphihatdxi3 = 0.0
        EKin = fd.assemble( 0.5 * (1.0/FfJacobian) * ( (Ffdxdxi3**2+Ffdzdxi3**2)*(Ffdpsidxi0*phihat+varphimp.dx(0))**2 \
                                    -2.0*(Ffdxdxi*Ffdxdxi3+Ffdzdxi*Ffdzdxi3)*(Ffdpsidxi0*phihat+varphimp.dx(0))*(Ffdpsidxi3*dphihatdxi3+varphimp.dx(1)) \
                                    + (Ffdxdxi**2+Ffdzdxi**2)*(Ffdpsidxi3*dphihatdxi3+varphimp.dx(1))**2 ) * fd.dx )
        EPot = fd.assemble( gg*Ffdxdxi*( 0.5*fd.inner(Ffz,Ffz)-H0*Ffz+0.5*H0**2) * fd.ds_t )
        EKin = EKin
        EPot = EPot
    elif nvpcase == 24:
        EKin = fd.assemble( 0.5 * ( (Lw**2/Ww) * (H0+fac*etamp) * (phimp.dx(0)+varphimp.dx(0)-(z/(H0+fac*etamp))*fac*etamp.dx(0)*(facc*phimp.dx(1)+varphimp.dx(1)))**2 \
                                     + Ww*(H0**2/(H0+fac*etamp)) * (faccc*phimp.dx(1)+varphimp.dx(1))**2 ) * fd.dx )
        EPot = fd.assemble( (gg*Ww*H0*(0.5*fd.inner(H0+etamp, H0+etamp)-(H0+etamp)*H0+0.5*H0**2) ) * fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
        Ekb = 0.5*mm*Wwmp**2
        Epb = mm*gg*Zzmp
    # End if
        
    E = EKin+EPot
        
    plt.figure(2)
    if nvpcase == 24:
        Ebuoy = Ekb+Epb
        Ewave = E
        E = Ewave+Ebuoy
        plt.plot(t,Ekb,'.g')
        plt.plot(t,Epb,'.c')
        plt.plot(t,Ebuoy,'.k')
        plt.plot(t,Ewave,'.k')
        plt.plot(t,E,'.k')
        plt.plot(t,EPot,'.b')
        plt.plot(t,EKin,'.r')       
        plt.ylabel(f'$E(t), K(t) (r), P(t) (b), e(t), k(t) (g), p(t) (c)$',fontsize=size)
    else:
        plt.plot(t,E,'.k')
        plt.plot(t,EPot,'.b')
        plt.plot(t,EKin,'.r')
        plt.ylabel(f'$E(t), K(t), P(t)$',fontsize=size)
    # End if  
    plt.xlabel(f'$t$ [s]',fontsize=size)

    if nvpcase == 22: # VP nonlinear steps 1 and 2 combined
        phi_f.assign(phiii)
        eta.assign(eta_new)
    elif nvpcase == 23: # VP nonlinear steps 1 and 2 combined
        phi_f.interpolate(2.0*phimp-phi_f)
        eta.interpolate(2.0*etamp-eta)
    elif nvpcase == 233: # VP nonlinear steps 1 and 2 combined
        phi_f.interpolate(2.0*phimp-phi_f)
        eta.interpolate(2.0*etamp-eta)
        Ffz = x[1]*etamp/H0
        Rwave.assign((1-nowaveflap)*Rwavemaker(t+dt,gam,sigm,tstop)) 
        Wflaps.assign(nowaveflap*Wflap(t+dt,gam,sigm,tstop))
        phii.interpolate(phi_f/(1.0-( Rwave+Ffz*Wflaps )/Lw))
        varphi.interpolate(varphimp+phi_f)
    elif nvpcase == 24: # VP nonlinear steps 1 and 2 combined with buoy
        phi_f.interpolate(2.0*phimp-phi_f)
        eta.interpolate(2.0*etamp-eta)
        Zz.interpolate(2.0*Zzmp-Zz)
        Wz.interpolate(2.0*Wzmp-Wz)
    
    t+= dt
    if (t in t_plot): # if (t >= tmeet-0.5*dt): # t > tmeet-epsmeet
        print('Plotting starts')
        plt.figure(1)
        i += 1
        tmeet = tmeet+dtmeet

        eta1vals = np.array([eta.at(x, zslice) for x in xvals])
        if nvpcase == 111: # VP linear
            phi1vals = np.array([phii.at(x, zslice) for x in xvals]) #phi1vals = np.array([phi.at(x, zslice) for x in xvals])
        elif nvpcase == 2: # VP nonlinear
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        elif nvpcase == 21: # VP nonlinear
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        elif nvpcase == 25: # VP nonlinear; waveflap; for plotting only
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        elif nvpcase == 233: # VP nonlinear; waveflap; for plotting only
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        else: # 22, 23, 233, 24, phi_f at surface so fine
            phi1vals = np.array([phi_f.at(x, zslice) for x in xvals])

        if nic == 0:
            ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
            ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')
            phi_exact_exprv = D * np.cos(kx * xvals) * np.cosh(kx * H0) * np.sin(omega * t) #
            eta_exact_exprv = A * np.cos(kx * xvals) * np.cos(omega * t)
            # KOKI: maybe use different markers to distinguish solutions at different times?
            ax1.plot(xvals, eta_exact_exprv, '-c', linewidth=1) # 
            ax2.plot(xvals, phi_exact_exprv, '-c', linewidth=1) #
            ax1.legend(loc=4)
            ax2.legend(loc=4)
            print('t =', t, tmeet, i)
        elif nic == 1:
            if t>=Tstartmeas: # t >= 0*tstop:
                ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
                ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')
                ax2.legend(loc=4)
                print('t =', t, tmeet, i) # ax1.legend(loc=4) # ax2.legend(loc=4)
                if nvpcase == 24: # Plot buoy reference level & line from (Lx,Zz-Keel) to (Lx-xss,Zz-Keel+Slopebuoy*(Lx-xss)) for xss = 1.5*(H0-Keel-Zz)/Slopebuoy)
                    ax1.plot(Lx,Zz, 'xk', linewidth=2) #
                    xss = 1.5*(H0-Keel-Zz)/Slopebuoy
                    Zxss = Zz-Keel+Slopebuoy*(Lx-xss)
                    ax1.plot([xss,Lx],[Zxss,Zz], '-k', linewidth=2) #
                    ax1.legend(loc=4)
                    # ax2.legend(loc=4) # output_data()
        if nvpcase==22 or nvpcase == 23 or nvpcase==233 or nvpcase==23:
            outfile_eta.write(eta, time=t)
            outfile_phi.write(phi_f, time=t)
            if nvpcase == 23 or nvpcase==233 or nvpcase==23:
                outfile_varphi.write(varphi, time=t)
        else:
            outfile_eta.write(eta0, time=t)
            outfile_phi.write(phi0, time=t)
     
    # print('t=',t)

# toc = time.time() - tic
print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
