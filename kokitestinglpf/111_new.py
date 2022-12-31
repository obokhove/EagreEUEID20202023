#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 13:41:59 2022

@author: mmwr
"""

import firedrake as fd
import math
import numpy as np
import matplotlib.pyplot as plt
import os



nvpcase = 2 

g = 9.81  # gravitational acceleration [m/s^2]

# water
Lx = 20.0  # length of the tank [m] in x-direction; needed for computing initial condition
Lz = 10.0  # height of the tank [m]; needed for computing initial condition
H0 = Lz # rest water depth [m]

nx = 200
nz = 2

# control parameters
output_data_every_x_time_steps = 20  # to avoid saving data every time step

save_path =  "lin_pot_flow" 
if not os.path.exists(save_path):
    os.makedirs(save_path)                                       

top_id = 4
#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
size = 16  # font size of image axes
factor = 0
t = 0
tt = format(t, '.3f') 

#________________________ MESH  _______________________#

mesh = fd.RectangleMesh(nx, nz, Lx, Lz)
x,z = fd.SpatialCoordinate(mesh)

xvals = np.linspace(0, Lx-10**(-10), nx)
zvals = np.linspace(0, Lz-10**(-10), nz) # ONNO 07-12 why -0.001 and not at top?
zslice = H0
xslice = 0.5*Lx

# The equations are in nondimensional units, hence we 
L = 1
T = 1
Lx /= L
Lz /= L
gg = g

## initial condition in fluid based on analytical solution
## compute analytical initial phi and eta
n_mode = 2

kx = 2 * np.pi * n_mode / Lx
omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
A = 0.2
D = -gg*A/(omega*np.cosh(kx*H0))

Tperiod = 2*np.pi/omega
print('Period: ', Tperiod)

x = mesh.coordinates

t0 = 0.0
phi_exact_expr = D * fd.cos(kx * x[0]) * fd.cosh(kx * x[1]) * np.sin(omega * t0) # D cos(kx*x) cosh(kx*z) cos(omega t)
eta_exact_expr = A * fd.cos(kx * x[0]) * np.cos(omega * t0)

t_end = Tperiod  # time of simulation [s]
dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
Nt = 2*100 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
dt = Tperiod/Nt  # 0.005  # time step [s]
dt = dtt
print('dtt=',dtt, t_end/dtt,dt,2/omega)


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
print('t_plot', t_plot,nt,nplot, t_end)
print('gg:',gg)
color = np.array(['g-', 'b--', 'r:', 'm:'])
colore = np.array(['k:', 'c--', 'm:'])


##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

fig, (ax1, ax2) = plt.subplots(2)

# ONNO 18-12 removed: ax2.set_title(r'$\phi$ value in $x$ direction',fontsize=tsize)
if nvpcase == 0:
    ax1.set_title(r'Weak form used:',fontsize=tsize)
elif nvpcase == 1:
    ax1.set_title(r'Functional derivative VP used:',fontsize=tsize)
elif nvpcase == 11:
    ax1.set_title(r'Functional derivative VP used, steps 1 & 2:',fontsize=tsize)
elif nvpcase == 111:
    ax1.set_title(r'Functional derivative VP used, steps 1 & 2:',fontsize=tsize)
elif nvpcase == 2:
    ax1.set_title(r'VP nonlinear case used:',fontsize=tsize)
# end if
ax1.set_ylabel(r'$\eta (x,t) [m]$ ',fontsize=size)
ax1.grid()
ax2.set_xlabel(r'$x [m]$ ',fontsize=size)
ax2.set_ylabel(r'$\phi (x,t)$ ',fontsize=size)
ax2.grid()


#__________________  Define function spaces  __________________##

nCG = 1
V_W = fd.FunctionSpace(mesh, "CG", nCG)

phi = fd.Function(V_W, name="phi")
varphi = fd.Function(V_W, name="varphi")
phi_new = fd.Function(V_W, name="phi")

phi_f = fd.Function(V_W, name="phi_f")  # at the free surface
phif_new = fd.Function(V_W, name="phi_f") 

eta = fd.Function(V_W, name="eta")
eta_new = fd.Function(V_W, name="eta_new")

heta = fd.Function(V_W, name="heta")
h_new = fd.Function(V_W, name="h_new")

trial_W = fd.TrialFunction(V_W)
trial_eta = fd.TrialFunction(V_W)
trial_phi = fd.TrialFunction(V_W)
v_W = fd.TestFunction(V_W)

mixed_V = V_W * V_W
mphi, mvarphi = fd.Function(mixed_V)


trial_eta, trial_varphi = fd.TrialFunctions(mixed_V)
del_eta, del_varphi = fd.TestFunctions(mixed_V)

result_mixed = fd.Function(mixed_V)
vvp = fd.TestFunction(mixed_V)

vvp0, vvp1 = fd.split(vvp)  # These represent "blocks".

# vvp0, vvp1 = fd.TestFunctions(mixed_V)

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

def varphi_BC():
    V_varphi = fd.FunctionSpace(mesh, "CG", nCG)
    # varphi.assign(1.0)
    bc = fd.DirichletBC(V_varphi , 0, top_id)
    f = fd.Function(V_varphi , dtype=np.int32)
    ## f is now 1 everywhere, except on the boundary
    bc.apply(f)
    return MyBC(V_varphi, 0, f)

def varphi_BC_mixed():
        bc_mixed = fd.DirichletBC(mixed_V.sub(0), 0, top_id)
        f_mixed = fd.Function(mixed_V.sub(0), dtype=np.int32)
        bc_mixed.apply(f_mixed)
        return MyBC(mixed_V.sub(0), 0, f_mixed)

def surface_BC_mixed():
        bc_mixed = fd.DirichletBC(mixed_V.sub(0), 1, top_id)
        f_mixed = fd.Function(mixed_V.sub(0), dtype=np.int32)
        bc_mixed.apply(f_mixed)
        return MyBC(mixed_V.sub(0), 0, f_mixed)

BC_exclude_beyond_surface = surface_BC() # ONNO 08-12 do not understand what is inside: surface_BC() or MyBC()
                                         # KOKI 16-12 I think the name is confusing; should be BC_exclude_below_surface?
                                         
BC_excluder_free_surface = varphi_BC()                            
                                         
BC_exclude_beyond_surface_mixed = surface_BC_mixed()                                         
BC_phi_f = fd.DirichletBC(V_W, phi_f, top_id)
BC_phif_new = fd.DirichletBC(V_W, phif_new, top_id)

BC_phi = fd.DirichletBC(V_W, phi, top_id)

# varphi.assign(1.0)
# BC_varphi = fd.DirichletBC(varphi, 0, top_id)
BC_varphi = fd.DirichletBC(V_W, 0, top_id)
BC_varphi_mixed = fd.DirichletBC(mixed_V.sub(1), 0, top_id)

BC_phi_new = fd.DirichletBC(V_W, phi_new, top_id)

## ____________ ICs ________________ ##
# tmp_eta, tmp_phi = result_mixed.split()

bc_top = fd.DirichletBC(V_W, 0, top_id)
eta.assign(0.0)
phi.assign(0.0)
eta_exact = fd.Function(V_W)
eta_exact.interpolate(eta_exact_expr)
eta.assign(eta_exact, bc_top.node_set)

phi.interpolate(phi_exact_expr)
phi_f.assign(phi, bc_top.node_set)


if nvpcase==111: # ONNO 19-12: above case 11 but with combo step for steps 1 and 2 solved in tandem? As test for nonlinear case 2?

    # VP11 = ( fd.inner(phi, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * gg * fd.inner(eta, eta)) )* fd.ds(top_id) \
    #     - ( 1/2 * fd.inner(fd.grad(phi+varphi), fd.grad(phi+varphi))  ) * fd.dx
        
    VP11 = ( fd.inner(phif_new, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * gg * fd.inner(eta, eta)) )* fd.ds(top_id) \
           - ( 1/2 * fd.inner(fd.grad(phif_new + varphi), fd.grad(phif_new  + varphi))  ) * fd.dx  
           
    # Step-1 and 2 must be solved in tandem: f-derivative VP wrt eta to find update of phi at free surface

    phif_expr1 = fd.derivative(VP11, eta, du=vvp0)  # du=v_W represents perturbation # 23-12 Make du split variable
 #
    # Step-2: f-derivative VP wrt varphi to get interior phi given sruface update phi
    phi_expr1 = fd.derivative(VP11, varphi, du=vvp1) # 23-12 Make du split variable

    Fexpr = phif_expr1 + phi_expr1
    
    # phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixed, bcs = [BC_phi, BC_varphi]))  
    phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixed, bcs = [BC_exclude_beyond_surface_mixed, BC_varphi_mixed]))

    # Step-3: f-derivative wrt phi but restrict to free surface to find updater eta_new; only solve for eta_new by using exclude
    # eta_expr2 = fd.derivative(VP11, phi, du=v_W)
    eta_expr2 = fd.derivative(VP11, phif_new, du=v_W)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr2, eta_new, bcs=BC_exclude_beyond_surface))
    
      
elif nvpcase == 0:
    print('case 111 with explicit weak forms is solved each step separartely')
    
    Fexpr = v_W * ( (phif_new - phi_f)/dt  + gg *  eta ) * fd.ds(top_id) 
    phif_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr , phif_new, bcs= BC_exclude_beyond_surface))
    
    print(' Step 1 completed')

    Fexpr1 = ( fd.inner(fd.grad(varphi), fd.grad(v_W)) + fd.inner( fd.grad(phif_new), fd.grad(v_W) ) ) * fd.dx
    phi_expr1 = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr1 , varphi,  bcs = BC_varphi))
    print(' Step 2 completed')
    
    eta_expr = v_W *  (eta_new - eta)/dt * fd.ds(top_id) - ( fd.dot(fd.grad(varphi), fd.grad(v_W)) + fd.inner( phif_new.dx(1), v_W.dx(1)) )* fd.dx
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new, bcs= BC_exclude_beyond_surface ))
    print(' Step 3 completed')
    
elif nvpcase == 1:
     print('case 111 with explicit weak forms is solved with step 1 and 2 combined')
             
     Fexpr = del_eta * ( (phif_new - phi_f)/dt  + gg *  eta ) * fd.ds(top_id)\
             + ( fd.inner(fd.grad(varphi), fd.grad(del_varphi)) + fd.inner( fd.grad(phif_new), fd.grad(del_varphi) ) ) * fd.dx
             
     phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixed, bcs = [BC_exclude_beyond_surface_mixed, BC_varphi_mixed]))
     
     eta_expr = v_W *  (eta_new - eta)/dt * fd.ds(top_id) - ( fd.dot(fd.grad(varphi), fd.grad(v_W)) + fd.inner( phif_new.dx(1), v_W.dx(1)) )* fd.dx
     eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr, eta_new, bcs= BC_exclude_beyond_surface ))
     
     
elif nvpcase == 2:
    print('case 111 with fd.derivative')
    
    VP11 = ( fd.inner(phif_new, (eta_new - eta)/dt) + fd.inner(phi_f, eta/dt) - (1/2 * gg * fd.inner(eta, eta)) )* fd.ds(top_id) \
           - ( 1/2 * fd.inner(fd.grad(phif_new + varphi), fd.grad(phif_new  + varphi))  ) * fd.dx    

    phif_expr1 = fd.derivative(VP11, eta)
    phi_expr1 = fd.derivative(VP11, varphi)
    
    Fexpr = phif_expr1 + phi_expr1
    
    # phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr , phif_new, bcs= [BC_exclude_beyond_surface, BC_varphi] ))
    phi_combo = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr , phif_new, bcs= BC_exclude_beyond_surface))

    # Step-3: f-derivative wrt phi but restrict to free surface to find updater eta_new; only solve for eta_new by using exclude
    eta_expr2 = fd.derivative(VP11, phif_new, du=v_W)
    eta_expr = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(eta_expr2, eta_new, bcs=BC_exclude_beyond_surface))
##_____________ 

bc_top = fd.DirichletBC(V_W, 0, top_id)
eta.assign(0.0)
phi.assign(0.0)

varphi.assign(0.0)
eta_exact = fd.Function(V_W)
eta_exact.interpolate(eta_exact_expr)
eta.assign(eta_exact, bc_top.node_set)
phi.interpolate(phi_exact_expr)
phi_f.assign(phi, bc_top.node_set)

# varphi.assign(phi, bc_top.node_set)
varphi.assign(phi)

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

if nvpcase == 111:
    EKin = fd.assemble( 0.5*fd.inner(fd.grad(phi+varphi),fd.grad(phi+varphi))*fd.dx )
    EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds(top_id) )
else:
    EKin = fd.assemble( 0.5*fd.inner(fd.grad(phi_f + varphi),fd.grad(phi_f + varphi))*fd.dx )
    EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds(top_id) )
E = EKin+EPot
E0 = E
plt.figure(2)
# plt.plot(t,E-E0,'.')
plt.plot(t,E,'.k')
plt.plot(t,EPot,'.b')
plt.plot(t,EKin,'.r')
plt.xlabel(f'$t$')
plt.ylabel(f'$E(t)$')


plt.title(r'Functional derivative VP used steps 1 & 2:',fontsize=tsize)
print('Plotting starts, initial data:')
plt.figure(1)
eta1vals = np.array([eta.at(x, zslice) for x in xvals])
phi1vals = np.array([phi.at(x, zslice) for x in xvals])            

ax1.plot(xvals, eta1vals, ':k', label = f' $\eta_n: t = {t:.3f}$',linewidth=2)
ax2.plot(xvals, phi1vals, ':k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)



output_data()
print('Time Loop starts')

while t <= t_end + dt:
    # print("time = ", t * T)
    # symplectic Euler scheme
    tt = format(t, '.3f') 
    
    if nvpcase==111:
        phi_combo.solve() #  ONNO 19-12 TODO mix variable?
        # phi, varphi = result_mixed.split()
        phif_new, varphi = result_mixed.split()
        eta_expr.solve()
        
    elif nvpcase == 0:
        phif_expr.solve()
        phi_expr1.solve()
        eta_expr.solve()
        
    elif nvpcase == 1:
         phi_combo.solve() 
         phif_new, varphi = result_mixed.split()
         eta_expr.solve()
         
    elif nvpcase == 2:
        phi_combo.solve()
        eta_expr.solve()
        
    t +=  dt
    
    if nvpcase == 111:
        phif_new, varphi = result_mixed.split()
        phi_f.assign(phif_new)
        # varphi.assign(varphi)
        eta.assign(eta_new)
        
    elif nvpcase == 0:
        phi_f.assign(phif_new)
        varphi.assign(varphi) # doing this update has no effect on the solution
        eta.assign(eta_new)
        
    elif nvpcase == 1:
        phif_new, varphi = result_mixed.split()
        phi_f.assign(phif_new)
        varphi.assign(varphi)
        eta.assign(eta_new)
    
    elif nvpcase == 2:
        phi_f.assign(phif_new)
        eta.assign(eta_new)
        
    if (t in t_plot):
        print('Plotting starts')
        plt.figure(1)
        print('t =', t,i)
        i += 1
        
        eta1vals = np.array([eta.at(x, zslice) for x in xvals])
        phi1vals = np.array([phi_f.at(x, zslice) for x in xvals])            
        
        ax1.plot(xvals, eta1vals, color[int(i-1) % 4], label = f' $\eta_n: t = {t:.3f}$')
        ax2.plot(xvals, phi1vals, color[int(i-1) % 4], label = f' $\phi_n: t = {t:.3f}$')

        # Free-surface exact expressions
        phi_exact_exprv = D * np.cos(kx * xvals) * np.cosh(kx * H0) * np.sin(omega * t) #
        eta_exact_exprv = A * np.cos(kx * xvals) * np.cos(omega * t)

        # KOKI: maybe use different markers to distinguish solutions at different times?
        # ax1.plot(xvals, eta_exact_exprv, ':', linewidth=1,label = f' $\eta_e: t = {t:.3f}$' ) # ONNO 18-12 still does not look to converge with nvpcase == 0,1; wrong exact solution or at wrong time?
        # ax2.plot(xvals, phi_exact_exprv, ':', linewidth=1, label = f' $\eta_e: t = {t:.3f}$') # ONNO 18-12 still does not look to converge with nvpcase == 0,1; wrong exact solution or at wrong time?
        
        ax1.legend(loc=4)
        ax2.legend(loc=4)
        output_data()


    if nvpcase == 111:
            EKin = fd.assemble( 0.5*fd.inner(fd.grad(phi+varphi),fd.grad(phi+varphi))*fd.dx )
            EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds(top_id) )
    else:
        EKin = fd.assemble( 0.5*fd.inner(fd.grad(phi_f + varphi),fd.grad(phi_f + varphi))*fd.dx )
        EPot = fd.assemble( 0.5*gg*fd.inner(eta,eta)*fd.ds(top_id) )
        
    E = EKin+EPot
    plt.figure(2)
    plt.plot(t,E,'.k')
    plt.plot(t,EPot,'.b')
    plt.plot(t,EKin,'.r')
    plt.xlabel('$t$')
    plt.ylabel('$E(t)$')
plt.show() 
print('*************** PROGRAM ENDS ******************')