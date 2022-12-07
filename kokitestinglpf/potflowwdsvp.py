import firedrake as fd
import math
import numpy as np
import matplotlib.pyplot as plt
import os


# parameters in SI units
# t_end = 5.0  # time of simulation [s]
# dt = 0.005  # time step [s]
g = 9.8  # gravitational acceleration [m/s^2]

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

top_id = 4

nvpcase = 0 # ONNO: standard weak-form approach with 3 steps 1: VP approach with two steps

#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
size = 16 # font size of image axes
factor = 0
t = 0
tt = format(t, '.3f') 

#________________________ MESH  _______________________#

mesh = fd.RectangleMesh(nx, nz, Lx, Lz)
x,z = fd.SpatialCoordinate(mesh)

xvals = np.linspace(0, Lx-0.001  , nx)
zvals = np.linspace(0, Lz- 0.001  , nz) 
zslice = Lz
xslice = Lx/2

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
a = 0.0 * T / L ** 2  # in nondim units
b = 0.005 * T / L ** 2  # in nondim units

kx = 2 * np.pi * n_mode / Lx
omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
A = 1.0
D = -gg*A/(omega*np.cosh(kx*H0))

Tperiod = 2*np.pi/omega
print('Period: ', Tperiod)
x = mesh.coordinates
##phi_exact_expr = a * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) # Huh?
##eta_exact_expr = -omega * b * fd.cos(kx * x[0]) * fd.cosh(kx * Lz) # Huh?

t0 = 0.0
phi_exact_expr = D * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
eta_exact_expr = A * fd.cos(kx * x[0]) * np.cos(omega * t0)

t_end = Tperiod  # time of simulation [s]
dtt = Lx/nx/np.sqrt(gg*H0)
Nt = 200 # 500 poor 5000 better, 1st order poor
dt = Tperiod/Nt  # 0.005  # time step [s]
print('dtt=',dtt, t_end/dtt,dt)


##______________  To get results at different time steps ______________##

time = []
t = 0
while (t <= t_end+dt):
        time.append(t)
        t+= dt

nplot = 4
t2 = int(len(time)/2)
nt = int(len(time)/nplot)
t_plot = np.array([ time[0], time[t2], time[-1] ])
t_plot = time[0::nt]
print('t_plot', t_plot,nt,nplot)
print('gg:',gg)
color = np.array(['g-', 'b--', 'r:'])
colore = np.array(['k:', 'c--', 'm:'])


##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

fig, (ax1, ax2) = plt.subplots(2)

ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)
ax1.set_title(r'$\eta$ value in $x$ direction',fontsize=tsize)
ax1.set_ylabel(r'$\eta (x,t) [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
ax2.grid()


#__________________  Define function spaces  __________________##

V_W = fd.FunctionSpace(mesh, "CG", 1)

phi = fd.Function(V_W, name="phi")
phi_new = fd.Function(V_W, name="phi")

phi_f = fd.Function(V_W, name="phi_f")  # at the free surface
phif_new = fd.Function(V_W, name="phi_f") 

eta = fd.Function(V_W, name="eta")
eta_new = fd.Function(V_W, name="eta")

trial_W = fd.TrialFunction(V_W)
v_W = fd.TestFunction(V_W)

mixed_V = V_W * V_W
trial_eta, trial_phi = fd.TrialFunctions(mixed_V)
del_eta, del_phi = fd.TestFunctions(mixed_V)
result_mixed = fd.Function(mixed_V)

# KOKI 06-12-2022
test_mixed = fd.TestFunction(mixed_V)



##_________________  Boundary Conditions __________________________##

class MyBC(fd.DirichletBC):
    def __init__(self, V, value, markers):
        super(MyBC, self).__init__(V, value, 0)
        self.nodes = np.unique(np.where(markers.dat.data_ro_with_halos == 0)[0])


def surface_BC():
    bc = fd.DirichletBC(V_W, 1, top_id)
    f = fd.Function(V_W, dtype=np.int32)
    ## f is now 0 everywhere, except on the boundary
    bc.apply(f)
    return MyBC(V_W, 0, f)


BC_exclude_beyond_surface = surface_BC()
BC_phi_f = fd.DirichletBC(V_W, phi_f, top_id)
BC_phif_new = fd.DirichletBC(V_W, phif_new, top_id)

BC_phi = fd.DirichletBC(V_W, phi, top_id)

BC_phi_new = fd.DirichletBC(V_W, phi_new, top_id)

# 
# ONNO 01-12
# Working equations format:
#
if nvpcase == 0:
    # Step-1: update phi at free surface using explicit forward Euler on eta but impliciy on phi (nonlinear case so not here) 
    phif_expr =  v_W * ((phi - phi_f)/dt  + gg * eta ) * fd.ds(top_id) # g ? derivative of VP wrt eta^n+1 to get the value of phi^n+1b at top surface
    phif_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phif_expr, phi, bcs = BC_exclude_beyond_surface))
    # ONNO 06-12 error phi_f is used at top not phif_new? BC_phi_f = fd.DirichletBC(V_W, phif_new, top_id)

    # Step-2: Update phi_new in interior using the updated phi as bc. 
    phi_expr = fd.dot(fd.grad(phi_new), fd.grad(v_W)) * fd.dx
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr, phi_new, bcs = BC_phi))
    # phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi1_expr, phi, bcs = BC_phi_f)) # ONNO 06-12 old one Wajiha; how is this using the top updated phi?
    # ONNO 07-12: How do I know wjether bcs = BC_phi uploads the new phi at the free surface solved in Step-1?
    
    # Step-3: update eta_new at free surface using all updated phi_new (which now includes updated phi at free surface from Step-1) backward Euler step
    eta_expr = v_W *  (eta_new - eta)/dt * fd.ds(top_id) - fd.dot(fd.grad(phi_new), fd.grad(v_W)) * fd.dx
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new, bcs = BC_exclude_beyond_surface ))
    #
elif nvpcase==1:
    # KOKI 06-12
    VP =  ( fd.inner(phi, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * g * fd.inner(eta, eta)) )* fd.ds(top_id) \
        + ( - 1/2 * fd.inner(fd.grad(phi), fd.grad(phi))  ) * fd.dx
    phif_expr1 = fd.derivative(VP, eta, du=v_W)  # du=v_W represents perturbation
    phif_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phif_expr1, phi, bcs=BC_exclude_beyond_surface))
    # ONNO TRY OLD NORMAL stuff
    #phi1_expr = fd.dot(fd.grad(phi), fd.grad(v_W)) * fd.dx
    #phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi1_expr, phi, bcs = BC_phi))
    # eta_expr1 = v_W *  (eta_new - eta)/dt * fd.ds(top_id) - fd.dot(fd.grad(phi_new), fd.grad(v_W)) * fd.dx
    # eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr1, eta_new, bcs= BC_exclude_beyond_surface ))
    phi_expr1 = fd.derivative(VP, phi, du=test_mixed) # IN HERE no BC has been applied yet
    #  Issue is that bsc=BC_phi removes the free surface rows while they should be kept and put on RHS.
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr1, result_mixed, bcs = BC_phi)) 
    eta, phi = split(result_mixed)
    # so only solve for phi or have a fake/wrong solve for eta.

    # Third step here where we use again the variational derivative wrt phi but now solve for eta only whilst using tyhe new phin from the previous two steps
    # Ensure that phi is the new phi from step 2
    eta_expr2 = fd.derivative(VP, phi, du=test_mixed)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr2,eta))
    # only solve for eta
    #
elif nvpcase==2:
    # 
    # ONNO 01-12
    # Desired VP format of the above
    # 
    # VP formulation of above with phi^n+1=phi_f at free top surface the same but interior phi^(n+1) and surface eta^(n+1) in one go
    #
    VP =  ( fd.inner(trial_phi, (trial_eta - eta)/dt) + fd.inner(phi, eta/dt) - (1/2 * g * fd.inner(eta,eta)) )* fd.ds(top_id) \
        + ( - 1/2 * H0 * fd.inner(fd.grad(trial_phi), fd.grad(trial_phi))  ) * fd.dx

    # 
    # derivative of VP wrt eta=eta^n to get the value of trial_phi=phi^n+1 at top surface only
    phif_expr1 = fd.derivative(VP, eta, v_W)  # ONNO 01-12 don't understand why v_W and not V_W 
    # phif_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phif_expr1, trial_phi, bcs= BC_exclude_beyond_surface)) # ONNO 02-12-2022: FAILS
    phif_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phif_expr1, trial_phi, bcs= BC_exclude_beyond_surface)) # ONNO 02-12-2022: FAILS

    # 
    # derivative of VP wrt trial_phi=phi^n+1 to get the value of trial_phi=phi^n+1 in interior
    #                                                        and trial_eta=eta^(n+1) at surface in one go using solved phi at top.
    phi_expr1 = fd.derivative(VP, trial_phi, v_W)  
    phi_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(phi_expr1, result_mixed, bcs = BC_phi_f))
    #
# end if

#  
# tmp_eta, tmp_phi = result_mixed.split()

bc_top = fd.DirichletBC(V_W, 0, top_id)
eta.assign(0.0)
phi.assign(0.0)
eta_exact = fd.Function(V_W)
eta_exact.interpolate(eta_exact_expr)
eta.assign(eta_exact, bc_top.node_set)
phi.interpolate(phi_exact_expr)
phi_f.assign(phi, bc_top.node_set)


###### OUTPUT FILES ##########
outfile_phi = fd.File("results/phi.pvd")
outfile_eta = fd.File("results/eta.pvd")


def output_data():
    output_data.counter += 1
    if output_data.counter % output_data_every_x_time_steps != 0:
        return
    mesh_static = mesh.coordinates.vector().get_local()
    mesh.coordinates.dat.data[:, 1] += eta.dat.data_ro
    
    outfile_eta.write( eta_new )
    outfile_phi.write( phi_new )
    mesh.coordinates.vector().set_local(mesh_static)
    
output_data.counter = -1  # -1 to exclude counting print of initial state

t = 0.0
i = 0.0

print('Plotting starts, initial data:')
eta1vals = np.array([eta.at(x, zslice) for x in xvals])
phi1vals = np.array([phi_f.at(x, zslice) for x in xvals])            

ax1.plot(xvals, eta1vals, ':k', label = f' $\eta_n: t = {t:.3f}$',linewidth=2)
ax2.plot(xvals, phi1vals, ':k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)
# ONNO 07-12: something is off since t~0 inthe time loop is not close to the initial condition. phi-solve seems wrong; dotted black lines are IC.
        


output_data()
print('Time Loop starts')

while t <= t_end + dt:
    # print("time = ", t * T)
    # symplectic Euler scheme
    tt = format(t, '.3f') 

    if nvpcase == 0:
        # normal weak-form case
        phif_expr.solve() # solves phi^(n+1) at top free surface
        # ONNO 07-12 not needed? phi_f.assign(phi)
        # ONNO 06-12 needed?        phi_f.assign(phif_new) now solved with new BSC case above
        phi_expr.solve() # solves phi^(n+1) in interior
        eta_expr.solve() # solves eta^(n+1) at top free surface
    elif nvpcase == 1:
        # ONNO: to do: VP solve
        phif_expr.solve() # solves phi^(n+1) at top free surface same as above
        phi_expr.solve() # solves phi^(n+1) in interior and eta^(n+1) at top surface simulataneously
        #  ONNO 06-12 cheat old stuff:
        eta_expr.solve() # solves eta^(n+1) at top free surface
        #  mixed variables needs to be assignd ONNO to do?
        #  ONNO is this right:
        # ONNO 06-12 commented out eta_new, phi_new = result_mixed.split() # ONNO: issue is that now only nterior has beemn assigned; how is phi_new at top assigned? Automatically or not?
    # end if

    if nvpcase == 0:
        phi_f.assign(phif_new)
        phi_new.assign(phi)
        eta.assign(eta_new)
    elif nvpcase == 1:  #  ONNO new; # is this right? ONNO:
        phi.assign(phi_new)
        eta.assign(eta_new)
    # end if

    if (t in t_plot):
        print('Plotting starts')
        print('t =', t,i)
        i += 1
        
        eta1vals = np.array([eta_new.at(x, zslice) for x in xvals])
        phi1vals = np.array([phi.at(x, zslice) for x in xvals])            
        
        ax1.plot(xvals, eta1vals, color[int(i-1) % 3], label = f' $\eta_n: t = {t:.3f}$')
        ax2.plot(xvals, phi1vals, color[int(i-1) % 3], label = f' $\phi_n: t = {t:.3f}$')
        
        ax1.legend(loc=4)
        ax2.legend(loc=4)
        output_data()
     
    t+= dt


plt.show() 
print('*************** PROGRAM ENDS ******************')
