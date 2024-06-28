import firedrake as fd
from firedrake import (
    min_value
)
import math
from math import *
import time as tijd
import numpy as np
import sympy as sp
from sympy import summation
from sympy.abc import k
import matplotlib.pyplot as plt
import os
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule
# import os.path
plt.close("all")

# parameters in SI units

# Parameters coil
D_rad = 0.0002769      # % radius of coil MW30-8
a_rad = 0.012          # % coil outer radius was 0.04 now 24mm diameter
Hm = 0.2               # % mast length
Lm = 0.025           # % magnet length was 0.04 now 25mm
Am = 0.0075         # % magnet radius was 0.8*a_rad diamter 15mm
L = Lm             # % length of coils: 2 coils in Faraday shaking lamp twice as long as 2 magnets
mu0 = 4*pi*10**(-7)    # % permeability of vacuum
nli = 10               # nli layers
N = nli*L/D_rad        # % winding number
M0 = 777.1237          # % magnetisation density (chosen such that m=0.1; too low by factor 100; see website below)
m = M0*pi*Am**2*Lm     # % magnetic dipole moment
m = 10 #  https://www.quadrant.us/en/magnetic-calculator-web 50mm long 15mm diameter, N52 material: 10 Am ^2 8836 mm^3 density 7400 0.065 kilogram; 65 grams
                       # https://www.magnetsource.com/pages/neodymium-magnets
muu = mu0*m/(4*pi)     
alp = 0.05             # % correction parameter in G(Z)
Zbar = 0.18            # % height of buoy at rest
Mm = 0.08 # Mass buoy plus magnet plus mast
kspring = 4*pi*Mm
print('kspring:', kspring)
Kf = 0.53
Li = Kf*np.pi*a_rad**2*mu0*N**2/L # coil induction
nq = 1
Vt = 2.05
Isat = 0.02
Rl = nq*Vt/Isat
fac3  = 1.0
Rl1 = fac3*Rl
sigmm = 5.96*10**7
Rc = 8*a_rad*N/(sigmm*D_rad**2)
Ri = Rc
R = Rc+Rl+Ri
R1 = Rc+fac3*Rl+Ri
H0 = 0.1
Hk = 0.04
Ly = 2
Lc = 0.2508
Lx = 0.2
Lp = 0.0
grav = 9.8 

thetac = np.pi*68.26/180 # to check with Jonny's code thetac = np.pi*45/180

Lc2 = Ly-0.5*Lx*np.tan(thetac)
Lc = Lc2

tan_th = (Ly-Lc)/(0.5*Lx)

print('2x tthetac Lc ',np.arctan(tan_th),thetac,Lc)
rho0 = 997
alp3 = 0.5389316 #  np.arctan(3*Mm*tan_th/rho0/(Ly-Lp)**3)

Zbar2 = H0+Hk-(3*Mm*np.tan(thetac)*np.tan(alp3)**2/rho0)**(1/3)
Zbar = Zbar2
print('Rl Rc Zbar Zbar2 H0 Hk Lc2 combo',Rl,Rc, Zbar, Zbar2, H0, Hk,Lc2,3*Mm*np.tan(thetac)*np.tan(alp3)/rho0)
print('Mm,np.tan(thetac),alp3,rho0)', Mm,np.tan(thetac),alp3,rho0) 
gamma = 2*pi*a_rad**2*muu*N/L
print('gamma, Rc, Rl, Ri Li m',gamma,Rc,Rl,Ri,Li,m)

# Approximated G(Z)
def Gapprox(Zz,Lb,Le,a_rad,Hm,alp,L,Zbar):
    GaZ = 1.0/(a_rad**2 + (Zbar + alp*Hm - Zz +Lb)**2)**(3/2) - 1.0/(a_rad**2 + (Zbar + alp*Hm - Zz + Le)**2)**(3/2)
    return GaZ
Z00 = 0.09
Z00 = Zbar+0.03
Z00 = Zbar
Z00g = Z00
# Z00 = 0.205
GZ0 = Gapprox(Z00,-L/2,L/2,a_rad,Hm,alp,L,Zbar)
print('Zbar Z00, GZ0, gamma GZ0',Zbar,Z00,GZ0,gamma*GZ0)
# zze =  sp.Symbol('zze')
# bbb =  sp.Symbol('bbb')
# sp.integrate(Gapprox(zze,-L/2,L/2,a_rad,Hm,alp,L,Zbar),zze,(0,bbb))
plt.figure(1)
Nz1 = 200
ZZmin = 0.0
ZZmax = 0.2
ZZ1 = np.linspace(ZZmin,ZZmax,Nz1)
plt.plot(ZZ1,gamma*Gapprox(ZZ1,-L/2,L/2,a_rad,Hm,alp,L,Zbar),'-')
plt.plot(Z00,gamma*Gapprox(Z00,-L/2,L/2,a_rad,Hm,alp,L,Zbar),'xr')

# water
Lx = 10.0 # 
nx = 1   # 

# control parameters
output_data_every_x_time_steps = 20  # to avoid saving data every time step

save_path =  "lin_pot_flow" 
if not os.path.exists(save_path):
    os.makedirs(save_path)                                       

top_id = 'top'


#__________________  FIGURE PARAMETERS  _____________________#

tsize = 18 # font size of image title
tsize2 = 12
size = 16  # font size of image axes
factor = 0
t = 0
tt = format(t, '.3f') 

#________________________ MESH  _______________________#
# 

mesh = fd.IntervalMesh(nx, Lx)
x = fd.SpatialCoordinate(mesh)

xvals = np.linspace(0.0, Lx-10**(-10), nx)
xslice = 0.5*Lx

## initial condition nic=1

x = mesh.coordinates

t0 = 0.0
nic = 0
nvpcase = 2 # MMP: 1: linear 1-coil model; 2: nonlinear 1-coil model; 3: linear 3-coil model; 4: nonlinear 3-coil model
if nvpcase == 1: # linear 1-coil model
    nic = 1
elif nvpcase == 2: # nonlinear 1-coil model
    nic = 1

time = []
t = 0
    
if nic == 1:
    # Parameters time stepping and such
    Tperiod = 1.0
    nTfac = 2*8
    nforcefac = 2*6
    nforcefac2 = 2*7
    t_end = nTfac*Tperiod # time of simulation [s]
    Tstartmeas = 0.0
    dtt = np.minimum(0.01,0.005) # i.e.
    dtt = 0.05
    Nt = 1 # 
    CFL = 0.5
    dt = CFL*dtt # CFL*dtt
    print('dtt=',dtt, t_end/dtt)
    nplot = 400
    dtmeet = t_end/nplot # 
    tmeet = dtmeet
    tmeas = np.linspace(0.0, t_end, nplot+1)
    tstop = nTfac-1
    
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 400
    if nvpcase==2:
        Z00g = Zbar
        Zc = Zbar # Z00
        Wc = 0.0
        Ic = 0
        Qc = 0
        
    
    
dtmeet = t_end/nplot # 
tmeet = dtmeet
tmeas = np.linspace(0.0, t_end, nplot+1)
print(' dtmeet, tmeet', dtmeet, tmeet)
print('tmeas', tmeas)
epsmeet = 10.0**(-10)
nt = int(len(time)/nplot)
print('nt',nt,len(time))
t_plot = time[0::nt]
#print('t_plot', t_plot, nt, nplot, t_end)


##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
# 
ax1.set_title(r'Forced 1-coil buoy-wire model',fontsize=tsize2)   
ax1.set_ylabel(r'$Z(t)$ ',fontsize=size)
ax2.set_ylabel(r'$W(t)$ ',fontsize=size)
ax3.set_ylabel(r'$Q(t)$ ',fontsize=size)
ax4.set_ylabel(r'$I(t)$ ',fontsize=size)
ax5.set_ylabel(r'$P_g(t)$ ',fontsize=size)
ax5.set_xlabel(r'$t$ ',fontsize=size)


#__________________  Define function spaces  __________________#

nCG = 1
nDG = 0
V_C = fd.FunctionSpace(mesh, 'DG', nDG, vfamily='DG') # 
V_CC = fd.FunctionSpace(mesh, 'R', 0, vfamily='R', vdegree=0) # buoy variables Z and W

# Variables for modified midpoint (dry) buoy-generator model hanging from a spring with moving hanging point
mixed_Vmpc = V_C * V_C
result_mixedmpc = fd.Function(mixed_Vmpc)
vvmpc = fd.TestFunction(mixed_Vmpc)
vvmpc0, vvmpc1 = fd.split(vvmpc) # These represent "blocks".
Q12, Z12 = fd.split(result_mixedmpc) # Only variations needed
W12, Ic12 = fd.split(result_mixedmpc) # But for completeness these enter the VP

# Charge Q, current Ic, position Z and velocity W
# Two test function for W12 and Ic12 rest elminated by hand
Z0 = fd.Function(V_C)
Z0g = fd.Function(V_C)
Z1 = fd.Function(V_C)
W0 = fd.Function(V_C)
W1 = fd.Function(V_C)
Q0 = fd.Function(V_C)
Q1 = fd.Function(V_C)
I0 = fd.Function(V_C)
I1 = fd.Function(V_C)
vw11 = fd.TestFunction(V_C)
vic12 = fd.TestFunction(V_C)


##_________________  Initial Conditions __________________________##

if nvpcase==1:
    Z0g.assign(Z00g)
    Z0.assign(Zc)
    W0.assign(Wc)
    Q0.assign(Qc)
    I0.assign(Ic)
    t = 0.0
    i = 0.0
    print('t, Zc, Wc',t,Zc,Wc,Qc,Ic)
    Z00 = np.array(Z0.vector())
    W00 = np.array(W0.vector())
    Q00 = np.array(Q0.vector())
    Ic00 = np.array(I0.vector())
elif nvpcase==2:
    Z0g.assign(Z00g)
    Z0.assign(Zc)
    W0.assign(Wc)
    Q0.assign(Qc)
    I0.assign(Ic)
    t = 0.0
    i = 0.0
    print('t, Zc, Wc',t,Zc,Wc,Qc,Ic)
    Z00 = np.array(Z0.vector())
    W00 = np.array(W0.vector())
    Q00 = np.array(Q0.vector())
    Ic00 = np.array(I0.vector())
ax1.plot(t,Z00,'.')
ax2.plot(t,W00,'.')
ax3.plot(t,Q00,'.')
ax4.plot(t,Ic00,'.')

# 
#
#
if nvpcase==2: # MMP 1-coil nonlinear case
    vpolyp = 5

    Amp = 0.0653
    facA = 0.4
    Amp = facA*Amp
    sigm = 2.0*np.pi/Tperiod
    
    def Rwavemaker(t,Amp,sigm,tstop):
        Rh1 = Amp*fd.cos(sigm*t)
        if t >= tstop:
            Rh1 = 0.0*Amp*fd.cos(sigm*tstop) # check whether should be 0
        return Rh1
    def dRwavemakerdt(t,Amp,sigm,tstop):
        Rt1 = -Amp*sigm*fd.sin(sigm*t)         
        if t >= tstop:
            Rt1 = -0.0*Amp*sigm*fd.sin(sigm*tstop)
        return Rt1
    force12 = fd.Constant(0.0)
    force12.assign(Rwavemaker(t+0.5*dt,Amp,sigm,tstop)) 
    force12dt = fd.Constant(0.0)
    force12dt.assign(dRwavemakerdt(t+0.5*dt,Amp,sigm,tstop))

    GZZ12 = 1.0/(a_rad**2 + (Zbar + alp*Hm - (Z0+0.5*dt*(W12+force12dt)) -L/2)**2)**(3/2) - 1.0/(a_rad**2 + (Zbar + alp*Hm - (Z0+0.5*dt*(W12+force12dt)) + L/2)**2)**(3/2)
    # linear GZZ12 = 1.0/(a_rad**2 + (Zbar + alp*Hm - (Zbar) -L/2)**2)**(3/2) - 1.0/(a_rad**2 + (Zbar + alp*Hm - (Zbar) + L/2)**2)**(3/2)


    noption = 1
    if noption==1: # works
        rfac = 0.0
        wfac = 1.0
    elif noption==2: # does not converge correctly
        rfac = 1.0
        wfac = 0.0
    elif noption==3: # works
        rfac = 0.0
        wfac = 1.0
    VPnl = (1/Lx)*( Mm*W12*(Z1-Z0)/dt - Mm*Z12*(W1-W0)/dt - Mm*W12*force12dt - 0.5*Mm*W12**2 - rfac*0.5*kspring*(Z12-Zbar-force12)**2 + (Li*Ic12)*(Q1-Q0)/dt - Q12*Li*(I1-I0)/dt - 0.5*Li*Ic12**2 )*fd.dx(degree=vpolyp)
    W_expr = fd.derivative(VPnl, Z12, du=vvmpc0) # du=v_R eqn for W1
    # W_expr = W_expr + (1/Lx)*( -vvmpc0*gamma*GZZ12*((Q1-Q0)/dt) )*fd.dx(degree=vpolyp) # Add buoy-generator coupling term
    # W_expr = fd.replace(W_expr, {Q1: Q0+dt*Ic12}) # Replace Q1
    if noption==1:
        W_expr = W_expr + (1/Lx)*( -vvmpc0*(gamma*GZZ12*Ic12 + wfac*kspring*(Z0+0.5*dt*(W12+force12dt)-Zbar-force12)) )*fd.dx(degree=vpolyp) # Add buoy-generator coupling term
    elif noption==2:
        W_expr = W_expr + (1/Lx)*( -vvmpc0*(gamma*GZZ12*Ic12) )*fd.dx(degree=vpolyp) # Add buoy-generator coupling term
        W_expr = fd.replace(W_expr, {Z12: Z0+0.5*dt*(W12+force12dt)}) # NOTE OB 26-06: Replace Z12 not allowed. Why? Because ito W12?
    W_expr = fd.replace(W_expr, {W1: 2*W12-W0}) # W1 = 2*Wh12-W0
    
    smallfac = 10**(-6)
    cc = nq*Vt
    tfac = 1.0 # 0: linear damping ; 1: nonlinear damping
    lfac = 0.0 # 1: linear damping ; 0: nonlinear damping
    ass = 1/Isat
    AbsI = fd.conditional(Ic12>0,Ic12,-Ic12)
    I_expr = fd.derivative(VPnl, Q12, du=vvmpc1) # du=v_R eqn for I1
    # I_expr = I_expr - (1/Lx)*( vvmpc1*( (Rc+Ri+lfac*Rl)*Ic12 ) ) *fd.dx(degree=vpolyp)
    I_expr = I_expr - (1/Lx)*( vvmpc1*((Rc+Ri+lfac*Rl)*Ic12 + tfac*fd.conditional(AbsI<smallfac,cc*ass*Ic12*(1-0.5*ass*AbsI+(ass*AbsI)**2/3-0.25*(ass*AbsI)**3+0.2*(ass*AbsI)**4),(cc*Ic12*fd.ln(1+ass*AbsI)/AbsI))  ) ) *fd.dx(degree=vpolyp)
    I_expr = I_expr + (1/Lx)*( vvmpc1*gamma*GZZ12*((Z1-Z0)/dt) )*fd.dx(degree=vpolyp) # Add buoy-generator coupling term
    I_expr = fd.replace(I_expr, {Z1: Z0+dt*(W12+force12dt)})
    I_expr = fd.replace(I_expr, {I1: 2*Ic12-I0})

    # Weak form testing GZZ12 = GZ0; Z12 = Z0+0.5*dt*(W12+force12dt)
    if noption==3: # works; also above I_expr with W_expr below works
        W_expr = (1/Lx)*(vvmpc0*( -(W12-W0)-(0.5/Mm)*dt*kspring*(Z0+0.5*dt*(W12+force12dt)-Zbar-force12)-(0.5/Mm)*dt*gamma*GZZ12*Ic12) )*fd.dx(degree=vpolyp)
        I_expr = (1/Lx)*(vvmpc1*( -(Ic12-I0)+(0.5/Li)*dt*gamma*GZZ12*(W12+force12dt) - (0.5/Li)*dt*(Rc+Ri+lfac*Rl)*Ic12 - tfac*fd.conditional(AbsI<smallfac,cc*ass*Ic12*(1-0.5*ass*AbsI+(ass*AbsI)**2/3-0.25*(ass*AbsI)**3+0.2*(ass*AbsI)**4),(cc*Ic12*fd.ln(1+ass*AbsI)/AbsI)) ) )*fd.dx(degree=vpolyp)
    Fexpr = W_expr+I_expr

    solver_parameters6 = {
        'sns_type': 'newtonls',
        'sns_atol': 1e-19,
        'mat_type': 'aij'
    }
    # 'sns_type': 'newtonls',
    # 'mat_type': 'nest' w linear works; w nonlinear gives results but odd compared with SE. Nonlinear w K12 does not converge
    # 'mat_type': 'aij' w linear works; w nonlinear gives results but odd compared with SE. Nonlinear w K12 does not converge
    
    solve1coil_l = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexpr, result_mixedmpc),solver_parameters=solver_parameters6)   
#    
# End if
#  

print('Time Loop starts')
tic = tijd.time()

Pgmean = 0.0*np.array(Z0.vector())
Pgave0 = 0.0*np.array(Z0.vector())
Pgave = 0.0*np.array(Z0.vector())

while t <= 1.0*(t_end + dt): #

    if  nvpcase==2:
        force12.assign(Rwavemaker(t+0.5*dt,Amp,sigm,tstop)) 
        force12dt.assign(dRwavemakerdt(t+0.5*dt,Amp,sigm,tstop))
        solve1coil_l.solve()
        W12, Ic12 = fd.split(result_mixedmpc)
        W1.interpolate(2*W12-W0)
        I1.interpolate(2*Ic12-I0)
        Z1.interpolate(Z0+dt*(W12+force12dt))
        Q1.interpolate(Q0+dt*Ic12)                       
    # End if
    
    t+= dt
    # if (t in t_plot): # 
    # print('Plotting starts')
    #     i += 1
    tmeet = tmeet+dtmeet
    if nvpcase==1:
        Z00 = np.array(Z0.vector())
        W00 = np.array(W0.vector())
        Z11 = np.array(Z1.vector())
        W11 = np.array(W1.vector())
        Q00 = np.array(Q0.vector())
        I00 = np.array(I0.vector())
        Q11 = np.array(Q1.vector())
        I11 = np.array(I1.vector())
        Pg00 = (Rl*I00**2)
        Pg11 = (Rl*I11**2)
        Pgmean = Pgmean + dt*Pg11
        Pgave = Pgmean/t
        ax1.plot([t-dt,t],[Z00,Z11],'-k')
        ax2.plot([t-dt,t],[W00,W11],'-k')
        ax3.plot([t-dt,t],[Q00,Q11],'-k')
        ax4.plot([t-dt,t],[I00,I11],'-k')
        ax5.plot([t-dt,t],[Pgave0,Pgave],'-k')
        Pgave0 = Pgave
        Z0.assign(Z1)
        W0.assign(W1)
        Q0.assign(Q1)
        I0.assign(I1)
    elif nvpcase==2:
        Z00 = np.array(Z0.vector())
        W00 = np.array(W0.vector())
        Z11 = np.array(Z1.vector())
        W11 = np.array(W1.vector())
        Q00 = np.array(Q0.vector())
        I00 = np.array(I0.vector())
        Q11 = np.array(Q1.vector())
        I11 = np.array(I1.vector())
        Pg00 = (Rl*I00**2)
        Pg11 = (Rl*I11**2)
        Pgmean = Pgmean + dt*Pg11
        Pgave = Pgmean/t
        ax1.plot([t-dt,t],[Z00,Z11],'-k')
        ax2.plot([t-dt,t],[W00,W11],'-k')
        ax3.plot([t-dt,t],[Q00,Q11],'-k')
        ax4.plot([t-dt,t],[I00,I11],'-k')
        ax5.plot([t-dt,t],[Pgave0,Pgave],'-k')
        Pgave0 = Pgave
        Z0.assign(Z1)
        W0.assign(W1)
        Q0.assign(Q1)
        I0.assign(I1)
    # print('t =', t, tmeet, i) #
            
# End while time loop      


toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
# print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
