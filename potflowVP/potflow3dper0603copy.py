import firedrake as fd
import math
from math import *
import time
import numpy as np
import sympy as sp
from sympy import summation
from sympy.abc import k
import matplotlib.pyplot as plt
import os
from firedrake import *
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule
from FIAT.quadrature import GaussLegendreQuadratureLineRule
from finat.point_set import PointSet, GaussLegendrePointSet, GaussLobattoLegendrePointSet
from finat.quadrature import QuadratureRule, TensorProductQuadratureRule
os.environ["OMP_NUM_THREADS"] = "1"

# parameters in SI units REMIS error of polynomials
gg = 9.81  # gravitational acceleration [m/s^2]

# water domain and discretisation parameters
Lx = 140 # [m]
Ly = 2 # [m]
Lz = 1 # [m]
H0 = Lz # rest water depth [m]
nx = 280
ny = 2
nz = 6
nCG = 2 # function space order horizontal
nCGvert = 2 # function space order vertical
nvpcase = "SV" # MMP=modified midpoint VP time discretisation in case of making more cases; SV=Stormer-Verlet
nphihatz = "Unity" # "unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.

# control parameters
output_data_every_x_time_steps = 20  # to avoid saving data every time step

save_path =  "lin_pot_flow" 
if not os.path.exists(save_path):
    os.makedirs(save_path)                                       


#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
tsize2 = 12
size = 16  # font size of image axes
factor = 0
t = 0
tt = format(t, '.3f') 

""" ________________ Switches ________________ """
domain_type = "single"
if domain_type == "single":
    direction = "x"
    y_shift = 0


#________________________ MESH  _______________________#

mesh2d = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction=direction,
                               quadrilateral=True, reorder=None,distribution_parameters=None,diagonal=None,comm=COMM_WORLD)
mesh = fd.ExtrudedMesh(mesh2d, nz, layer_height=Lz/nz, extrusion_type='uniform')
x, y, z = fd.SpatialCoordinate(mesh)
x = mesh.coordinates
top_id = 'top'
# 

# For in-situ plotting
xvals = np.linspace(0.0, Lx-10**(-10), nx)
zvals = np.linspace(0.0, Lz-10**(-10), nz) # 
zslice = H0
xslice = 0.5*Lx
yslice = 0.5*Ly

# Choice initial condition
time = [] # time for measurements
t = 0
nic = "linearw" # choice initial condition
if nic=="linearw": # linear waves in x-direction dimensional
    t0 = 0.0
    n_mode = 2
    kx = 2 * np.pi * n_mode / Lx
    omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
    A = 0.1
    D = gg*A/(omega*np.cosh(kx*H0))
    Tperiod = 2*np.pi/omega
    print('Period: ', Tperiod)
    psi_exact_expr = D * fd.sin(kx * x[0]-omega * t0) * fd.cosh(kx * x[2])
    psi_exact_exprH0 = D * fd.sin(kx * x[0]-omega * t0) * fd.cosh(kx * H0)
    eta_exact_expr = A * fd.cos(kx * x[0]-omega * t0)
    btopoexpr = 0.0*psi_exact_expr
    dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
    Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
    CFL = 2.0 # run at a) 0.125 and b) 0.5*0.125
    dt = CFL*Tperiod/Nt  # 0.005  # time step [s]
    nTfac = 2
    t_end = nTfac*Tperiod  # time of simulation [s]

    print('dtt=',dtt, t_end/dtt,dt,2/omega)
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    
    nplot = 16*nTfac
    fac = 1.0 # Used to split h=H0+eta in such in way that we can switch to solving h (fac=0.0) or eta (fac=1.0)
elif nic=="SP1": # SP1 soliton periodic dimensional
    t0 = 0.0
    # to do
elif nic=="SP2": # SP2 solitons periodic dimensional; extend VP?
    t0 = 0.0
    # to do
elif nic=="SP3": # SP3 solitons periodic dimensional; extend VP?
    t0 = 0.0
    # to do

Nenergy = 1000
dtmenergy = t_end/Nenergy
tmeetenergy = dtmenergy
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
                       
fig, (ax1, ax2) = plt.subplots(2)
if nvpcase=="MMP":
    ax1.set_title(r'VP 3D nonlinear midpoint, x-periodic:',fontsize=tsize2)
else:
    ax1.set_title(r'VP 3D nonlinear midpoint, x-periodic:',fontsize=tsize2)
    
ax1.set_ylabel(r'$h(x,t) [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\psi(x,y,t)$ ',fontsize=size)
ax2.grid()

#__________________  Quadratures and define function spaces  __________________#

orders = [2*nCG, 2*nCGvert]  # horizontal and vertical
quad_rules = []
for order in orders:
    fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), order)
    # Check: # print(fiat_rule.get_points())     # print(fiat_rule.get_weights())
    point_set = GaussLobattoLegendrePointSet(fiat_rule.get_points())
    quad_rule = QuadratureRule(point_set, fiat_rule.get_weights())
    quad_rules.append(quad_rule)
quad_rule = TensorProductQuadratureRule(quad_rules)
                                  
V_W = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='CG', vdegree=nCGvert) # interior potential varphi; can mix degrees in hor and vert
V_R = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='R', vdegree=0) # free surface eta and surface potential phi extended uniformly in vertical: vdegree=0

psi_f = fd.Function(V_R, name="phi_f") # velocity potential at level n at free surface
psii = fd.Function(V_R, name="psii") # velocity potential at level n+1 at free surface                                  
h_old = fd.Function(V_R, name="h_old") # water depth old at level n
h_new = fd.Function(V_R, name="h_new") # water depth new at level n+1
btopo = fd.Function(V_R, name="btopo") # topography fixed in time
varphi = fd.Function(V_W, name="varphi") # total velocity potential

if nvpcase=="MMP":
    # Variables at midpoint for modified midpoint waves
    mixed_Vmp = V_R * V_R * V_W
    result_mixedmp = fd.Function(mixed_Vmp)
    vvmp = fd.TestFunction(mixed_Vmp)
    vvmp0, vvmp1, vvmp2 = fd.split(vvmp) # These represent "blocks".
    psimp, hmp, varphimp= fd.split(result_mixedmp)
elif nvpcase=="SV":
    v_R = fd.TestFunction(V_R)
    mixed_Vsv = V_R * V_W
    result_mixedsv = fd.Function(mixed_Vsv)
    vvsv = fd.TestFunction(mixed_Vsv)
    vvsv0, vvsv1, = fd.split(vvsv) # These represent "blocks".
    psisv, varphisv = fd.split(result_mixedsv)

# Initialise variables; projections on main variables at initial time
if nvpcase=="MMP":
    h_old.interpolate(H0*(fac-1.0)+eta_exact_expr)
    psi_f.interpolate(psi_exact_exprH0)
    btopo.interpolate(btopoexpr)
    BC_varphi_mixedmp = fd.DirichletBC(mixed_Vmp.sub(2), 0, top_id) # varphimp condition for modified midpoint
elif nvpcase=="SV":
    h_old.interpolate(H0*(fac-1.0)+eta_exact_expr)
    psi_f.interpolate(psi_exact_exprH0)
    btopo.interpolate(btopoexpr)    
    BC_varphi_mixedsv = fd.DirichletBC(mixed_Vsv.sub(1), 0, top_id) # varphisv condition for modified midpoint

##_________________  Boundary Conditions (for varphimp) __________________________##

class MyBC(fd.DirichletBC):
    def __init__(self, V, value, markers):
        super(MyBC, self).__init__(V, value, 0)
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])
def surface_BC():
    bc = fd.DirichletBC(V_R, 1, top_id)
    f = fd.Function(V_R, dtype=np.int32) # f is now 0 everywhere, except on the boundary
    bc.apply(f)
    return MyBC(V_R, 0, f)

mixed_V = V_R * V_W
def surface_BC_mixed(): # 
        bc_mixed = fd.DirichletBC(mixed_V.sub(0), 1, top_id)
        f_mixed = fd.Function(mixed_V.sub(0), dtype=np.int32)
        bc_mixed.apply(f_mixed)
        return MyBC(mixed_V.sub(0), 0, f_mixed)


if nvpcase=="MMP": # modfied midpoint for solving psimp, hmp, varphimp= fd.split(result_mixedmp)
    param_h       = {'ksp_converged_reason':None, 'pc_type': 'fieldsplit','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'}               
    param_psi     = {'ksp_converged_reason':None, 'ksp_type': 'preonly', 'pc_type': 'lu'}
    param_hat_psi = {'ksp_converged_reason':None, 'ksp_type': 'preonly', 'pc_type': 'lu'}
    parameters = {"mat_type": "matfree","snes_monitor": None,
                  # We'll use a non-stationary Krylov solve for the Schur complement, so we need to use a flexible Krylov method on the outside. ::
                  "ksp_type": "fgmres","ksp_gmres_modifiedgramschmidt": None,"ksp_monitor_true_residual": None,
                  # Now to configure the preconditioner:: "pc_type": "fieldsplit","pc_fieldsplit_type": "schur","pc_fieldsplit_schur_fact_type": "upper",
                  "pc_type": "fieldsplit","pc_fieldsplit_type": "multiplicative",
                  # we invert the psimp block with LU::
                  "fieldsplit_0_ksp_type": "preonly","fieldsplit_0_pc_type": "python","fieldsplit_0_pc_python_type": "firedrake.AssembledPC","fieldsplit_0_assembled_pc_type": "lu",
                  # and for hmp invert the schur complement inexactly using GMRES, preconditioned # with PCD. ::
                  "fieldsplit_1_ksp_type": "gmres","fieldsplit_1_ksp_rtol": 1e-4,"fieldsplit_1_pc_type": "python","fieldsplit_1_pc_python_type": "firedrake.PCDPC",
                  # We now need to configure the mass and stiffness solvers in the PCD # preconditioner.  For this example, we will just invert them with LU
                  # although of course we can use a scalable method if we wish. First the # mass solve::
                  "fieldsplit_1_pcd_Mp_ksp_type": "preonly","fieldsplit_1_pcd_Mp_pc_type": "lu",
                  # and the stiffness solve.::
                  "fieldsplit_1_pcd_Kp_ksp_type": "preonly", "fieldsplit_1_pcd_Kp_pc_type": "lu",
                  # Finally, we just need to decide whether to apply the action of the # pressure-space convection-diffusion operator with an assembled matrix
                  # or matrix free.  Here we will use matrix-free::
                  "fieldsplit_1_pcd_Fp_mat_type": "matfree",
                  # lu solver for varphimp
                  "fieldsplit_2_ksp_type": "preonly","fieldsplit_2_pc_type": "python","fieldsplit_2_pc_python_type": "firedrake.AssembledPC","fieldsplit_2_assembled_pc_type": "lu"}
    param_psi = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    Ww = Lx
    Lw = Lx
    if nphihatz=="GLL1":
        fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), nCGvert+1) # GLL
        zk = (H0*fiat_rule.get_points())
        phihat = fd.product( (x[1]-zk.item(kk))/(H0-zk.item(kk)) for kk in range(0,nCGvert-1,1) )
        dphihat = phihat.dx(1) # May not work and in that case specify the entire product: dpsidxi3 = psimp*phihat.dx(1)
    elif nphihatz=="Unity":
        phihat = 1.0
        dphihat = 0.0

    VP3dpf = (- H0*Ww*fd.inner(psimp, (h_new - h_old)/dt) \
              + H0*fd.inner(hmp, (Ww*psii - Ww*psi_f)/dt) \
              + H0*gg*Ww*( 0.5*fd.inner(fac*H0+hmp, fac*H0+hmp)-(fac*H0+hmp)*H0+0.5*H0**2 ) ) * fd.ds_t \
                + 0.5*( (Lw**2/Ww)*(fac*H0+hmp)*(psimp.dx(0)*phihat+varphimp.dx(0)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(0)+x[2]*hmp.dx(0))*(psimp*dphihat+varphimp.dx(2)))**2 \
                        + Ww*(fac*H0+hmp)*(psimp.dx(1)*phihat+varphimp.dx(1)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(1)+x[2]*hmp.dx(1))*(psimp*dphihat+varphimp.dx(2)))**2 \
                        + Ww*(H0**2/(fac*H0+hmp)) * (psimp*dphihat+varphimp.dx(2))**2 ) * fd.dx
    # VP3dpf = (- H0*Ww*fd.inner(psimp, (h_new - h_old)/dt) \
    #         + H0*fd.inner(hmp, (Ww*psii - Ww*psi_f)/dt) \
    #       + H0*gg*Ww*( 0.5*fd.inner(hmp, hmp)-hmp*H0+0.5*H0**2 ) ) * fd.ds_t \
    #     + 0.5*( (Lw**2/Ww)*hmp*(psimp.dx(0)*phihat+varphimp.dx(0)-(1/hmp)*(H0*btopo.dx(0)+x[2]*hmp.dx(0))*(psimp*dphihat+varphimp.dx(2)))**2 \
    #         + Ww*hmp*(psimp.dx(1)*phihat+varphimp.dx(1)-(1/hmp)*(H0*btopo.dx(1)+x[2]*hmp.dx(1))*(psimp*dphihat+varphimp.dx(2)))**2 \
    #         + Ww*(H0**2/hmp) * (psimp*dphihat+varphimp.dx(2))**2 ) * fd.dx

    #  Step-1: solve h^(n+1/2) wrt psi^(n+1/2)
    psif_exprnl1 = fd.derivative(VP3dpf, psimp, du=vvmp0) # du=v_W represents perturbation 
    psif_exprnl1 = fd.replace(psif_exprnl1, {psii: 2.0*psimp-psi_f})
    psif_exprnl1 = fd.replace(psif_exprnl1, {h_new: 2.0*hmp-h_old}) 

    #  Step-2: solve psi^(n+1/2) wrt hmp=h^(n+1/2)
    h_exprnl1 = fd.derivative(VP3dpf, hmp, du=vvmp1)
    h_exprnl1 = fd.replace(h_exprnl1, {psii: 2.0*psimp-psi_f})
    h_exprnl1 = fd.replace(h_exprnl1, {h_new: 2.0*hmp-h_old})

    #  Step-3: wrt varmp=varphi^(n+1/2) solve varmp=varphi^(n+1/2)
    phi_exprnl1 = fd.derivative(VP3dpf, varphimp, du=vvmp2)
    phi_exprnl1 = fd.replace(phi_exprnl1, {psii: 2.0*psimp-psi_f})
    phi_exprnl1 = fd.replace(phi_exprnl1, {h_new: 2.0*hmp-h_old}) 

    Fexprnl = psif_exprnl1+h_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=param_psi)
    # phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=parameters)
elif nvpcase=="SV": # Stormer-Verlet to to
    param_psi = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    Ww = Lx
    Lw = Lx
    if nphihatz=="GLL1":
        fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), nCGvert+1) # GLL
        zk = (H0*fiat_rule.get_points())
        phihat = fd.product( (x[1]-zk.item(kk))/(H0-zk.item(kk)) for kk in range(0,nCGvert-1,1) )
        dphihat = phihat.dx(1) # May not work and in that case specify the entire product: dpsidxi3 = psimp*phihat.dx(1)
    elif nphihatz=="Unity":
        phihat = 1.0
        dphihat = 0.0

    # Variables [psisv, varphisv], h_new, psii
    VP3dpf = (- H0*Ww*fd.inner(psisv, (h_new - h_old)/dt) + H0*Ww*fd.inner(psii, h_new/dt) - H0*Ww*fd.inner(psi_f, h_old/dt) \
              + 0.5*H0*gg*Ww*( 0.5*fd.inner(fac*H0+h_new, fac*H0+h_new)-(fac*H0+h_new)*H0+0.5*H0**2 ) \
              + 0.5*H0*gg*Ww*( 0.5*fd.inner(fac*H0+h_old, fac*H0+h_old)-(fac*H0+h_old)*H0+0.5*H0**2 )  ) * fd.ds_t \
                + 0.25*( (Lw**2/Ww)*(fac*H0+h_new)*(psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(0)+x[2]*h_new.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(fac*H0+h_new)*(psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(1)+x[2]*h_new.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(H0**2/(fac*H0+h_new)) * (psisv*dphihat+varphisv.dx(2))**2 \
                        + (Lw**2/Ww)*(fac*H0+h_old)*(psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(0)+x[2]*h_old.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(fac*H0+h_old)*(psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(1)+x[2]*h_old.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(H0**2/(fac*H0+h_old)) * (psisv*dphihat+varphisv.dx(2))**2  ) * fd.dx

    #  Step-1-2: solve psisv, varphisv variation wrt h_old (eta_old) and varphisv
    psif_exprnl1 = fd.derivative(VP3dpf, h_old, du=vvsv0) # du=v_W represents perturbation 
    phi_exprnl1 = fd.derivative(VP3dpf, varphisv, du=vvsv1)
    Fexprnl = psif_exprnl1+phi_exprnl1
    phi_combonlsv = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedsv, bcs = BC_varphi_mixedsv), solver_parameters=param_psi)
    
    #  Step-3: solve h_new=h^(n+1/2) variation wrt psi^(n+1/2)
    h_exprnl1 = fd.derivative(VP3dpf, psisv, du=v_R)
    h_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(h_exprnl1,h_new))

    #  Step-4: variation wrt h_new fsolve psii=psi^(n+1)
    psin_exprnl1 = fd.derivative(VP3dpf, h_new, du=v_R)
    phin_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(psin_exprnl1 ,psii))

# 
    

    
###### OUTPUT FILES and initial PLOTTING ##########
outfile_psi = fd.File("results/psi.pvd")
outfile_height = fd.File("results/height.pvd")
outfile_varphi = fd.File("results/varphi.pvd")

t = 0.0
i = 0.0

print('Plotting starts, initial data:')
plt.figure(1)
eta1vals = np.array([h_old.at(x, yslice, zslice) for x in xvals]) # 
phi1vals = np.array([psi_f.at(x, yslice, zslice) for x in xvals])

ax1.plot(xvals, eta1vals, ':k', label = f' $\eta_n: t = {t:.3f}$',linewidth=2)
ax2.plot(xvals, phi1vals, ':k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)


###### TIME LOOP ##########
print('Time Loop starts')
# tic = time.time()
while t <= t_end + dt: #  t_end + dt
    tt = format(t, '.3f')

    if nvpcase == "MMP": # VP MMP
        phi_combonl.solve()
        psimp, hmp, varphimp = result_mixedmp.split()
        psi_f.interpolate(2.0*psimp-psi_f) # update n+1 -> n
        h_old.interpolate(2.0*hmp-h_old) # update n+1 -> n
        varphi.interpolate(varphimp+psi_f) # total velociy potential for plotting
    elif nvpcase == "SV": # VP SV
        phi_combonlsv.solve()
        psisv, varphisv = result_mixedsv.split()
        h_exprnl.solve()
        phin_exprnl.solve()
        phi_f.assign(psii)
        # Done later since needed in energy EKin: h_old.assign(h_new)
        varphi.interpolate(varphisv+psi_f) # total velociy potential for plotting
        

    # Energy monitoring: bit too frequent reduce
    if t>=0.0:
        if nvpcase=="MMP":
            EKin = fd.assemble( 0.5*( (Lw**2/Ww)*(fac*H0+hmp)*(psimp.dx(0)*phihat+varphimp.dx(0)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(0)+x[2]*hmp.dx(0))*(psimp*dphihat+varphimp.dx(2)))**2 \
                                      + Ww*(fac*H0+hmp)*(psimp.dx(1)*phihat+varphimp.dx(1)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(1)+x[2]*hmp.dx(1))*(psimp*dphihat+varphimp.dx(2)))**2 \
                                      + Ww*(H0**2/(fac*H0+hmp))*(psimp*dphihat+varphimp.dx(2))**2 ) * fd.dx )
            EPot = fd.assemble( H0*gg*Ww*( 0.5*fd.inner(fac*H0+hmp,fac*H0+hmp)-(fac*H0+hmp)*H0+0.5*H0**2 ) * fd.ds_t )
        elif nvpcase=="SV":
            EKin = fd.assemble( 0.25*( (Lw**2/Ww)*(fac*H0+h_new)*(psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(0)+x[2]*h_new.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(fac*H0+h_new)*(psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(1)+x[2]*h_new.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(H0**2/(fac*H0+h_new)) * (psisv*dphihat+varphisv.dx(2))**2 \
                                       (Lw**2/Ww)*(fac*H0+h_old)*(psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(0)+x[2]*h_old.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(fac*H0+h_old)*(psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(1)+x[2]*h_old.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(H0**2/(fac*H0+h_old)) * (psisv*dphihat+varphisv.dx(2))**2  ) * fd.dx  )
            Epot = fd.assemble( 0.5*H0*gg*Ww*( 0.5*fd.inner(fac*H0+h_new, fac*H0+h_new)-(fac*H0+h_new)*H0+0.5*H0**2 ) \
                                + 0.5*H0*gg*Ww*( 0.5*fd.inner(fac*H0+h_old, fac*H0+h_old)-(fac*H0+h_old)*H0+0.5*H0**2 ) * fd.ds_t )
            
        Etot = EKin+EPot
        plt.figure(2)
        plt.plot(t,Etot,'.k')
        plt.plot(t,EPot,'.b')
        plt.plot(t,EKin,'.r')
        plt.xlabel(f'$t$',fontsize=size)
        plt.ylabel(f'$E(t), K(t), P(t)$',fontsize=size)
        
    if nvpcase == "SV": # VP SV
        h_old.assign(h_new)

    t+= dt
    if (t in t_plot): # if (t >= tmeet-0.5*dt): # t > tmeet-epsmeet
        print('Plotting starts')
        plt.figure(1)
        i += 1
        tmeet = tmeet+dtmeet

        if nvpcase == "MMP": # 
            phi1vals = np.array([psi_f.at(x, yslice, zslice) for x in xvals])
            eta1vals = np.array([h_old.at(x, yslice, zslice) for x in xvals])
            if nic == "linearw":
                ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
                ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')
                phi_exact_exprv = D * np.sin(kx * xvals-omega * t) * np.cosh(kx * H0)
                eta_exact_exprv = A * np.cos(kx * xvals-omega * t)
                ax1.plot(xvals, eta_exact_exprv, '-c', linewidth=1) # 
                ax2.plot(xvals, phi_exact_exprv, '-c', linewidth=1) #
                ax1.legend(loc=4)
                ax2.legend(loc=4)
            else:
                ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
                ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')
            print('t =', t, tmeet, i)
            #
            outfile_height.write(h_old, time=t)
            outfile_psi.write(psi_f, time=t)
            outfile_varphi.write(varphi, time=t)

            
print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
