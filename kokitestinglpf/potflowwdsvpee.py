import firedrake as fd
import math
import numpy as np
import matplotlib.pyplot as plt
import os


# parameters in SI units
# t_end = 5.0  # time of simulation [s]
# dt = 0.005  # time step [s]
g = 9.81  # gravitational acceleration [m/s^2]

# water
Lx = 20.0  # length of the tank [m] in x-direction; needed for computing initial condition
Lz = 10.0  # height of the tank [m]; needed for computing initial condition
H0 = Lz # rest water depth [m]

nx = 120
nz = 6

# control parameters
output_data_every_x_time_steps = 20  # to avoid saving data every time step

save_path =  "lin_pot_flow" 
if not os.path.exists(save_path):
    os.makedirs(save_path)                                       

top_id = 'top'

nvpcase = 111 # ONNO 06-01 cases 111 and 1 do not work yet

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

x,z = fd.SpatialCoordinate(mesh)

xvals = np.linspace(0.0, Lx-10**(-10), nx)
zvals = np.linspace(0.0, Lz-10**(-10), nz) # ONNO 07-12 why -0.001 and not at top?
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

## initial condition in fluid based on analytical solution
## compute analytical initial phi and eta
n_mode = 2
a = 0.0 * T / L ** 2    # in nondim units
b = 0.005 * T / L ** 2  # in nondim units

kx = 2 * np.pi * n_mode / Lx
omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
A = 0.2*4
D = -gg*A/(omega*np.cosh(kx*H0))

Tperiod = 2*np.pi/omega
print('Period: ', Tperiod)
x = mesh.coordinates
## phi_exact_expr = a * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) # ONNO: Huh?
## eta_exact_expr = -omega * b * fd.cos(kx * x[0]) * fd.cosh(kx * Lz) # ONNO: Huh?

t0 = 0.0
phi_exact_expr = D * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
phi_exact_exprH0 = D * fd.cos(kx * x[0]) * fd.cosh(kx * H0) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
eta_exact_expr = A * fd.cos(kx * x[0]) * np.cos(omega * t0)

t_end = Tperiod  # time of simulation [s]
dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
CFL = 0.125
dt = CFL*Tperiod/Nt  # 0.005  # time step [s]
#dt = dtt
print('dtt=',dtt, t_end/dtt,dt,2/omega)


##______________  To get results at different time steps ______________##

time = []
t = 0
while (t <= t_end+dt):
        time.append(t)
        t+= dt

nplot = 4
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
    ax1.set_title(r'Functional derivative VP used, varphi, steps 1+2, 3:',fontsize=tsize2)
elif nvpcase == 2:
    ax1.set_title(r'VP nonlinear case used:',fontsize=tsize2)
# end if
ax1.set_ylabel(r'$\eta (x,t) [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
ax2.grid()


#__________________  Define function spaces  __________________##

nCG = 1
V_W = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='CG', vdegree=nCG)
V_R = fd.FunctionSpace(mesh, 'CG', nCG, vfamily='R', vdegree=0)

# phi = fd.Function(V_W, name="phi")
varphi = fd.Function(V_W, name="varphi")

phi_f = fd.Function(V_R, name="phi_f")  # at the free surface
phif_new = fd.Function(V_R, name="phi_f") 

eta = fd.Function(V_R, name="eta")
eta_new = fd.Function(V_R, name="eta_new")

heta = fd.Function(V_R, name="heta")
h_new = fd.Function(V_R, name="h_new")

v_W = fd.TestFunction(V_W)
v_R = fd.TestFunction(V_R)

mixed_V = V_R * V_W
# phi, varphi = fd.Function(mixed_V)
result_mixed = fd.Function(mixed_V)
vvp = fd.TestFunction(mixed_V)
vvp0, vvp1 = fd.split(vvp)  # These represent "blocks".
phii, varphii = fd.split(result_mixed)


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
BC_varphi_mixed = fd.DirichletBC(mixed_V.sub(1), 0, top_id)
# bc0 = DirichletBC(W.sub(0), as_vector([0.0, -sin(5*x)]), 3)


# 
#
#
if nvpcase==111: # as 11 but steps 1 and 2 in combined solve; step 3 separately 
    VP11 = ( fd.inner(phii, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * gg * fd.inner(eta, eta)) )* fd.ds_t \
        - ( 1/2 * fd.inner(fd.grad(phii+varphii), fd.grad(phii+varphii))  ) * fd.dx
    
    # Step-1 and 2 must be solved in tandem: f-derivative VP wrt eta to find update of phi at free surface
    # int -phi/dt + phif/dt - gg*et) delta eta ds a=0 -> (phi-phif)/dt = -gg * eta
    phif_expr1 = fd.derivative(VP11, eta, du=vvp0)  # du=v_W represents perturbation # 23-12 Make du split variable

    # Step-2: f-derivative VP wrt varphi to get interior phi given surface update phi
    # int nabla (phi+varphi) cdot nabla delta varphi dx = 0
    phi_expr1 = fd.derivative(VP11, varphii, du=vvp1)
    Fexpr = phif_expr1+phi_expr1
    phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixed, bcs = [BC_exclude_beyond_surface_mixed,BC_varphi_mixed]))

    # 
    # Step-3: f-derivative wrt phi but restrict to free surface to find updater eta_new; only solve for eta_new by using exclude
    eta_expr2 = fd.derivative(VP11, phii, du=v_R)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr2,eta_new,bcs=BC_exclude_beyond_surface))
elif nvpcase==2: # Steps 1 and 2 need solving in unison
    # 
    # Desired VP format of the above
    param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu'}
    # 
    # VP formulation of above with phi^(n+1)=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    #
    phii, varphii = fd.split(result_mixed)
    Lw = 0.5*Lx
    Ww = Lw # Later wavemaker to be added # eta_new -> h_new and eta -> heta ; Nonlinear potential-flow VP:
    facc = 0.0
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
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixed, bcs = [BC_exclude_beyond_surface_mixed,BC_varphi_mixed]), solver_parameters=param_psi)

    #  Step-3: linear solve; 
    heta_exprnl2 = fd.derivative(VPnl, phii, du=v_W)
    heta_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(heta_exprnl2,eta_new,bcs=BC_exclude_beyond_surface))
    # 
# end if
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

###### OUTPUT FILES ##########
outfile_phi = fd.File("results/phi.pvd")
outfile_eta = fd.File("results/eta.pvd")

t = 0.0
i = 0.0



print('Plotting starts, initial data:')
plt.figure(1)
eta1vals = np.array([eta.at(x, zslice) for x in xvals])
#  pphi1vals = np.array([phii.at(xvals, zslice)])
phi1vals = np.array([phii.at(x, zslice) for x in xvals])            

ax1.plot(xvals, eta1vals, ':k', label = f' $\eta_n: t = {t:.3f}$',linewidth=2)
ax2.plot(xvals, phi1vals, ':k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)

# output_data()
if nvpcase == 111:
    EKin = fd.assemble( 0.5*fd.inner(fd.grad(phii+varphii),fd.grad(phii+varphii))*fd.dx )
    EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds_t )
elif nvpcase == 2:    
    EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phi.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*phi.dx(1))**2 + Ww * (H0**2/(H0+fac*eta)) * (phi.dx(1))**2) * fd.dx )
    EPot = fd.assemble( gg*Ww*H0*( 0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
    EKin = EKin/(Lw*H0)
    EPot = EPot/(Lw*H0)
    
E = EKin+EPot
E0 = E
plt.figure(2)
# plt.plot(t,E-E0,'.')
plt.plot(t,E,'.k')
plt.plot(t,EPot,'.b')
plt.plot(t,EKin,'.r')
plt.xlabel(f'$t$')
plt.ylabel(f'$E(t)$')
if nvpcase == 111:
    plt.title(r'Functional derivative VP used steps 1 & 2:',fontsize=tsize)
    # phi_expr.solve() # ?
elif nvpcase == 2:
    plt.title(r'VP nonlinear used:',fontsize=tsize)
    
# end if
print('E0=',E,EKin,EPot)


print('Time Loop starts')

while t <= t_end + epsmeet:
    # print("time = ", t * T)
    # symplectic Euler scheme
    tt = format(t, '.3f') 

    if nvpcase == 111:
        # solve of phi everywhere steps 1 and 2 combined
        phi_combo.solve() # 
        phii, varphii = result_mixed.split()
        eta_expr.solve()
    elif nvpcase == 2:
        # phif_exprnl.solve() # solves phi^(n+1) at top free surface same as above
        #  phi_exprnl.solve() # solves phi^(n+1) in interior and eta^(n+1) at top surface simulataneously
        phi_combonl.solve()
        phii, varphii = result_mixed.split()
        heta_exprnl.solve() 
    # end if

    if nvpcase == 111:  # VP linear steps 1 and 2 combined
        phi_f.assign(phii)
        # phi.assign(phii+varphii)
        eta.assign(eta_new)
    elif nvpcase == 2: # ONNO 19-12
        phi_f.assign(phii)
        # phi.assign(phii+varphii)
        eta.assign(eta_new)
    # end if
    # Energy monitoring:
    if nvpcase == 111:
        EKin = fd.assemble( 0.5*fd.inner(fd.grad(phii+varphii),fd.grad(phii+varphii))*fd.dx )
        EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds_t )
    elif nvpcase == 2:
        EKin = fd.assemble( 1/2 * ( (Lw**2/Ww) * (H0+fac*eta) * (phii.dx(0)+varphii.dx(0)-(z/(H0+fac*eta))*fac*eta.dx(0)*(facc*phii.dx(1)+varphii.dx(1)))**2 + Ww * (H0**2/(H0+fac*eta)) * (facc*phii.dx(1)+varphii.dx(1))**2) * fd.dx )
        EPot = fd.assemble( gg*Ww*H0*(0.5*fd.inner(H0+eta, H0+eta)-fd.inner(H0+eta,H0)+0.5*H0**2) * fd.ds_t )
        EKin = EKin/(Lw*H0)
        EPot = EPot/(Lw*H0)
    E = EKin+EPot
    plt.figure(2)
    plt.plot(t,E,'.k')
    plt.plot(t,EPot,'.b')
    plt.plot(t,EKin,'.r')
    plt.xlabel(f'$t$')
    plt.ylabel(f'$E(t)$')
    
    
    t+= dt
    if (t in t_plot):
        # if (t >= tmeet-0.5*dt): # t > tmeet-epsmeet
        
        print('Plotting starts')
        plt.figure(1)
        print('t =', t, tmeet, i)
        i += 1
        tmeet = tmeet+dtmeet

        eta1vals = np.array([eta.at(x, zslice) for x in xvals])
        if nvpcase == 111: #
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
            #phi1vals = np.array([phi.at(x, zslice) for x in xvals])
        elif nvpcase == 2: #
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        else: # 
            phi1vals = np.array([phii.at(x, zslice) for x in xvals])
        
        ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
        ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')

        # Free-surface exact expressions
        # phi_exact_expr = D * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
        # eta_exact_expr = A * fd.cos(kx * x[0]) * np.cos(omega * t0)
        phi_exact_exprv = D * np.cos(kx * xvals) * np.cosh(kx * H0) * np.sin(omega * t) #
        eta_exact_exprv = A * np.cos(kx * xvals) * np.cos(omega * t)

        # KOKI: maybe use different markers to distinguish solutions at different times?
        ax1.plot(xvals, eta_exact_exprv, '-c', linewidth=1) # ONNO 18-12 still does not look to converge with nvpcase == 0,1; wrong exact solution or at wrong time?
        ax2.plot(xvals, phi_exact_exprv, '-c', linewidth=1) # ONNO 18-12 still does not look to converge with nvpcase == 0,1; wrong exact solution or at wrong time?
        
        ax1.legend(loc=4)
        ax2.legend(loc=4)
        # output_data()
     
    # print('t=',t)


plt.show() 
print('*************** PROGRAM ENDS ******************')
