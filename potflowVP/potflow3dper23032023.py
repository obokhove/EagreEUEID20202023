#
# 3D potential-flow water-wave equations in x-periodic channel based on implemenation with VPs
# =================================================
# Onno Bokhove 01-03-2023 to 09-03-2023 with help by Junho Choi
#
# .. rst-class:: emphasis
#
#     This tutorial was contributed `Onno Bokhove <mailto:O.Bokhove@leeds.ac.uk>`__.
#
# Time-step choices: MMP and SV
#
# Initial conditions/tests: "linearw"=linear waves and "SP1" single soliton.
#
# Vertical structure function (e.g., for one vertical-element setting with nz=1): "unity" or "GLL"; GLL not tested yet in this code
#
#
import firedrake as fd
import math
from math import *
import time as tijd
import numpy as np
import sympy as sp
from sympy import summation
from sympy.abc import k
import matplotlib.pyplot as plt
import os
import os.path
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
nic = "linearw" # choice initial condition
nic = "SP2"  # single soliton SP1
nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; "SV"=Stormer-Verlet
nphihatz = "Unity" # "unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.
Lx = 140 # [m]
Ly = 2 # [m]
Lz = 1 # [m]
H0 = Lz # rest water depth [m]
nx = 280
ny = 2
nz = 6
nCG = 2 # function space order horizontal
nCGvert = 2 # function space order vertical
xx1 = 0.0
yy1 = 0.0
xx2 = Lx
yy2 = Ly

if nic=="SP1": # soliton decaying sufficiently fast to be taken as periodi using BL expansion in z
    xtilde = 10
    muu = 0.01
    H0 = 1.0
    eps = 1.0 # 0.01, 0.1 and 0.4 work
    Lx = 2*H0/np.sqrt(muu)*xtilde # twice as long as in BokhoveKalogirou2016 since their set-up is a bit close to boundaries
    Lz = H0*1.0 # not sure about this one; no scaled z used on BL?
    H0 = Lz
    nx = 280
    ny = 2
    nz = 6
    nCG = 2 # function space order horizontal
    nCGvert = 2 # function space order vertical
    nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; SV=Stormer-Verlet
    nphihatz = "Unity" # "unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.
elif nic=="SP2": # two-soliton case made periodic
    eps = 0.05
    muu = eps**2
    H0 = 1.0
    y2hat = 20
    y1hat = -y2hat
    Y2KPE = (eps/np.sqrt(muu))*(3/np.sqrt(2))**(2/3)*y2hat
    Ly = (H0/np.sqrt(muu))*(y2hat-y1hat)
    yy2 = (H0/np.sqrt(muu))*y2hat
    yy1 = (H0/np.sqrt(muu))*y1hat
    tau0 = 0.0
    k4 = (2/9)**(1/6)/(4*np.sqrt(eps)) # np.tan(thetaa)
    
    yhatstar = -(np.sqrt(muu)/eps)*(np.sqrt(2)/3)**(2/3)*np.log(2)/k4**2

    xx2hat = k4*np.sqrt(eps)*(3/np.sqrt(2))**(1/3)*y2hat  + np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*np.log(2)/k4
    
    X1KPE = k4**2*tau0-(1.0/k4)*np.arccosh( np.exp(k4**2*Y2KPE)-2*np.exp(-k4**2*Y2KPE) )
    X2KPE = k4**2*tau0+(1.0/k4)*np.arccosh( np.exp(k4**2*Y2KPE)-2*np.exp(-k4**2*Y2KPE) )
    x2hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X2KPE+(1/eps)*np.sqrt(0.5*muu/eps)*tau0)
    x1hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X1KPE+(1/eps)*np.sqrt(0.5*muu/eps)*tau0)
    xx1 = (H0/np.sqrt(muu))*x1hat
    xx2 = (H0/np.sqrt(muu))*x2hat
    Lx = xx2-xx1
    value = ((2/9)**(1/6)*0.25/(0.87013))**2
    print('Lx,Ly,x1hat,x2hat,y1hat,y2hat: ',Lx,Ly,x1hat,x2hat,y1hat,y2hat)
    print('xx2hat,yhatstar,k4: ',xx2hat,yhatstar,k4,value,np.log(2))
    nx = 200
    ny = 100
    nz = 4
    nCG = 1 # function space order horizontal
    nCGvert = 1 # function space order vertical
    nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; SV=Stormer-Verlet
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
                               quadrilateral=True, reorder=None,distribution_parameters=None,diagonal=None,comm=COMM_WORLD) # xx1, xx2, yy1, yy2
mesh = fd.ExtrudedMesh(mesh2d, nz, layer_height=Lz/nz, extrusion_type='uniform')
x, y, z = fd.SpatialCoordinate(mesh)
x = mesh.coordinates
top_id = 'top'
# 

# For in-situ plotting
xvals = np.linspace(0.0, Lx-10**(-10), nx)
yvals = np.linspace(0.0, Ly-10**(-10), ny)
zvals = np.linspace(0.0, Lz-10**(-10), nz) # 
zslice = H0
xslice = 0.5*Lx
yslice = 0.0
yslice12 = 0.5*Ly
yslice10 = Ly-10**(-10)

# Choice initial condition
time = [] # time for measurements
t = 0

fac = 1.0 # Used to split h=H0+eta in such in way that we can switch to solving h (fac=0.0) or eta (fac=1.0)
# u0 = 0.0 # periodicity factor; default 0
U0y = x[0]+x[1]
c0y = x[0]+x[1]
dU0dy = 0.0*U0y.dx(1)
dc0dy = 0.0*c0y.dx(1)
u0 = dU0dy
if nic=="linearw": # linear waves in x-direction dimensional
    t0 = 0.0
    n_mode = 2
    kx = 2 * np.pi * n_mode / Lx
    omega = np.sqrt(gg * kx * np.tanh(kx * Lz))
    A = 0.2
    D = gg*A/(omega*np.cosh(kx*H0))
    Tperiod = 2*np.pi/omega
    print('Period: ', Tperiod)
    psi_exact_expr = D * fd.sin(kx * x[0]-omega * t0) * fd.cosh(kx * x[2])
    psi_exact_exprH0 = D * fd.sin(kx * x[0]-omega * t0) * fd.cosh(kx * H0)
    eta_exact_expr = A * fd.cos(kx * x[0]-omega * t0)
    btopoexpr = 0.0*psi_exact_exprH0
    dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
    Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
    CFL = 1.0 # run at a) 0.125 and b) 0.5*0.125 2,0 for mmp! CFL=0.5 for SV time-step restriction!
    CFL = 0.25
    dt = CFL*Tperiod/Nt  # 0.005  # time step [s]
    nTfac = 2
    t_end = nTfac*Tperiod  # time of simulation [s]

    print('dtt=',dtt, t_end/dtt,dt,2/omega)
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    
    nplot = 16*nTfac
elif nic=="SP1": # SP1 soliton periodic dimensional
    t0 = 0.0
    cc = 0.5
    x0 = 0.5*Lx # Halfway again as in BokhoveKalogirou2016
    qq = 0.5*np.sqrt(cc*eps/muu)*((np.sqrt(muu)/H0)*(x[0]-x0)-((1+eps*cc/6))*(np.sqrt(muu*gg*H0)/H0)*t0) # argument used in soliton dimensonal
    eta_exact_expr = eps*H0*(cc/3.0)*fd.cosh(qq)**(-2)
    xx0 = 0
    qq0 = 0.5*np.sqrt(cc*eps/muu)*((np.sqrt(muu)/H0)*(xx0-x0)-((1+eps*cc/6))*(np.sqrt(muu*gg*H0)/H0)*t0) # argument used in soliton dimensonal at 0
    qqLx = 0.5*np.sqrt(cc*eps/muu)*((np.sqrt(muu)/H0)*(Lx-x0)-((1+eps*cc/6))*(np.sqrt(muu*gg*H0)/H0)*t0) # argument used in soliton dimensonal at Lx
    psiqq0_exact_exprH0 = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qq0) \
                                            +0.25*(eps*muu/H0**2)*((H0+eta_exact_expr)**2)*fd.sinh(qq0)/fd.cosh(qq0)**3 )
    psiLx_exact_exprH0 = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qqLx) \
                                            +0.25*(eps*muu/H0**2)*((H0+eta_exact_expr)**2)*fd.sinh(qqLx)/fd.cosh(qqLx)**3 )
    u0 = (psiLx_exact_exprH0-psiqq0_exact_exprH0)/Lx + 0.0*x[1]
    u0py = u0.dx(1)
    psi_exact_exprH0 = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qq) \
                                            +0.25*(eps*muu/H0**2)*((H0+eta_exact_expr)**2)*fd.sinh(qq)/fd.cosh(qq)**3 ) - u0*x[0] # that's it I think?
    btopoexpr = 0.0*psi_exact_exprH0 # no topography
    ttilde = 9.5
    t_end = ttilde*(H0/np.sqrt(gg*H0*muu)) # dimensionless BLE end-time of BokhoveKalogirou2016 times time-scaling factor
    Nt = 100
    dt = t_end/Nt # Somehow depends on soliton speed; so dx/U0 with U0=x/t dimensional from argument soliton; see dtt
    CFL = 0.1
    dtt = CFL*(Lx/nx)*6.0/((1+eps*cc)*np.sqrt(gg*H0))    
    print('dtt=',dt,dtt,t_end/dtt)
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 10
    # to do
elif nic=="SP2": # SP2 solitons periodic dimensional; extend VP? Define eta and phi
    t0 = 0.0
    # xx1, xx2, yy2=Ly
    Fx = np.sqrt(eps/muu)*(3/np.sqrt(2))*(1/3)
    Fy = (eps/np.sqrt(muu))*(3/np.sqrt(2))**(2/3)
    
    # xx1 used x[1]=y function ; can this be automated via general case? x[0]-0.5*Lx , x[1]-0.5*Ly ; 0...Lx=(x2-x1)
    xx11 = 0.0 # 0.0
    xx22 = Lx # Lx
    xs = -xx1
    ys = -yy1
    print('xx1, xx2, yy1, yy2', xx1, xx2, yy1, yy2)
    theta11 = -k4*( (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3) )*(xx11-xs-np.sqrt(gg*H0)*t0) + k4**2*( (eps/H0)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)
    theta41 =  k4*( (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3) )*(xx11-xs-np.sqrt(gg*H0)*t0) + k4**2*( (eps/H0)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)
    KK1 = k4*(fd.exp(theta41)+2*fd.exp(theta11+theta41)+fd.exp(theta11))
    KKX1 = k4**2*(fd.exp(theta41)-fd.exp(theta11))
    KKXX1 = k4**3*(fd.exp(theta41)+fd.exp(theta11))
    KKY1 = k4**3*(fd.exp(theta41)+4*fd.exp(theta11+theta41)+fd.exp(theta11))
    KKYY1 = k4**5*(fd.exp(theta41)+8*fd.exp(theta11+theta41)+fd.exp(theta11))
    KKXY1 = k4**2*KKX1 # k4**4*(fd.exp(theta41)-fd.exp(theta11))
    GX1 = 2*Fx**2*( k4**2*KKX1/KK1-3*KKXX1*KKX1/KK1**2+2*KKX1**3/KK1**3 )
    GY1 = 2*Fy**2*( k4**2*KKXY1/KK1-2*KKXY1*KKY1/KK1**2-KKX1*KKYY1/KK1**2+2*KKX1*(KKY1)**2/KK1**3 )
    etax1y = eps*H0*((4/3)**(1/3))*2.0*( KKXX1/KK1 -(KKX1/KK1)**2 )
    psix1y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1-0.5*muu*((H0+etax1y)/H0)**2*(GX1+GY1))
    # psix1y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1)

    # xx2 used x[1]=y function ; can this be automated via general case?
    theta12 = -k4*( (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3) )*(xx22-xs-np.sqrt(gg*H0)*t0) + k4**2*( (eps/H0)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)
    theta42 =  k4*( (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3) )*(xx22-xs-np.sqrt(gg*H0)*t0) + k4**2*( (eps/H0)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)
    KK2 = k4*(fd.exp(theta42)+2*fd.exp(theta12+theta42)+fd.exp(theta12))
    KKX2 = k4**2*(fd.exp(theta42)-fd.exp(theta12))
    KKXX2 = k4**3*(fd.exp(theta42)+fd.exp(theta12))
    KKY2 = k4**3*(fd.exp(theta42)+4*fd.exp(theta12+theta42)+fd.exp(theta12))
    KKYY2 = k4**5*(fd.exp(theta42)+8*fd.exp(theta12+theta42)+fd.exp(theta12))
    KKXY2 = k4**2*KKX2  #  (fd.exp(theta42)-fd.exp(theta12))
    GX2 = 2*Fx**2*( k4**2*KKX2/KK2-3*KKXX2*KKX2/KK2**2+2*KKX2**3/KK2**3 ) # do it via fd .dx(0) derivative 2x ; use replace?
    GY2 = 2*Fy**2*( k4**2*KKXY2/KK2-2*KKXY2*KKY2/KK2**2-KKX2*KKYY2/KK2**2+2*KKX2*(KKY2)**2/KK2**3 ) # do it via fd .dx(1) derivative 2x
    etax2y = eps*H0*((4/3)**(1/3))*2.0*( KKXX2/KK2 -(KKX2/KK2)**2 )
    psix2y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2-0.5*muu*((H0+etax2y)/H0)**2*(GX2+GY2))
    # psix2y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2)

    U0y = (psix2y-psix1y)/(xx2-xx1)
    c0y = (xx2*psix1y-xx1*psix2y)/(xx2-xx1)
    # c0y = psix1y-U0y*xx1
    U0yxc0y = U0y*x[0]+c0y # (U0y*x[0]+c0y)
    sicko = U0y*x[0]+c0y
    u0 = U0yxc0y.dx(0)
    u0py = U0yxc0y.dx(1)

    # X = (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3) )*(x[0]-np.sqrt(gg*H0)*t0) ; Y = ( (eps/H0)*(3/np.sqrt(2))**(2/3) )*x[1]
    theta1 = -k4*( (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3) )*(x[0]-xs-np.sqrt(gg*H0)*t0) + k4**2*( (eps/H0)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)
    theta4 =  k4*( (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3) )*(x[0]-xs-np.sqrt(gg*H0)*t0) + k4**2*( (eps/H0)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)
    KK = k4*(fd.exp(theta4)+2*fd.exp(theta1+theta4)+fd.exp(theta1))
    KKX = k4**2*(fd.exp(theta4)-fd.exp(theta1))
    KKXX = k4**3*(fd.exp(theta4)+fd.exp(theta1))
    KKY = k4**3*(fd.exp(theta4)+4*fd.exp(theta1+theta4)+fd.exp(theta1))
    KKYY = k4**5*(fd.exp(theta4)+8*fd.exp(theta1+theta4)+fd.exp(theta1))
    KKXY = k4**2*KKX # k4**4*(fd.exp(theta4)-fd.exp(theta1))
    GX = 2*Fx**2*( k4**2*KKX/KK-3*KKXX*KKX/KK**2+2*KKX**3/KK**3 )
    GY = 2*Fy**2*( k4**2*KKXY/KK-2*KKXY*KKY/KK**2-KKX*KKYY/KK**2+2*KKX*(KKY)**2/KK**3 )
    # KKXXX = k4**2*KKX # k4**4*(np.exp(theta4)-np.exp(theta1))
    # KKXYY = k4**2*KKXY # k4**6*(np.exp(theta4)-np.exp(theta1))
    eta_exact_expr = eps*H0*((4/3)**(1/3))*2.0*( KKXX/KK -(KKX/KK)**2 )
    psi_exact_exprH0 = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu*((H0+eta_exact_expr)/H0)**2*(GX+GY))-(U0y*x[0]+c0y)
    # psi_exact_exprH0 = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK)-(U0y*x[0]+c0y)
    sickofit =         ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu*((H0+eta_exact_expr)/H0)**2*(GX+GY))-(U0y*x[0]+c0y)
    # psi_exact_exprH0 = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK)
    # psi_exact_exprH0 = (2*KKX/KK)
    
    print('Hallo; muu:',muu)
    btopoexpr = 0.0*psi_exact_exprH0 # no topography
    ttilde = 0.001
    t_end = ttilde*(H0/np.sqrt(gg*H0*muu)) # dimensionless BLE end-time of BokhoveKalogirou2016 times time-scaling factor
    Nt = 10
    dt = t_end/Nt # Somehow depends on soliton speed; so dx/U0 with U0=x/t dimensional from argument soliton; see dtt
    CFL = 1.0
    cc = 1
    dtt = CFL*(Lx/nx)*6.0/((1+eps*cc)*np.sqrt(gg*H0))    
    print('dtt=',dt,dtt,t_end/dtt)
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 1
    print('Hallo2.')
    tijd.sleep(0.1)
    
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

psi_f = fd.Function(V_R, name="psi_f") # velocity potential at level n at free surface
sick = fd.Function(V_R, name="sick") # temp
U0c0sickos = fd.Function(V_R, name="U0c0sickos") # temp
psii = fd.Function(V_R, name="psii") # velocity potential at level n+1 at free surface
h_old = fd.Function(V_R, name="h_old") # water depth old at level n
h_new = fd.Function(V_R, name="h_new") # water depth new at level n+1
btopo = fd.Function(V_R, name="btopo") # topography fixed in time
varphi = fd.Function(V_W, name="varphi") # total velocity potential
U0cor = fd.Function(V_R, name="U0cor") # 
c0cor = fd.Function(V_R, name="c0cor") # 

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

if nic=="SP2":
    U0cor.interpolate(U0y)
    c0cor.interpolate(c0y)
    sick.interpolate(psi_exact_exprH0)
    U0c0sickos.interpolate(sicko)
    # sick.interpolate(sickofit)

    
if nvpcase=="MMP": # modfied midpoint for solving psimp, hmp, varphimp= fd.split(result_mixedmp)
    param_h       = {'ksp_converged_reason':None, 'pc_type': 'fieldsplit','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'}               
    param_psi1     = {'ksp_converged_reason':None, 'ksp_type': 'preonly', 'pc_type': 'lu'} # works 1 iteration; collapsed after 1 iteration even quicker than 46 at A=1, CFL=2
    param_psi2     = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'pc_type': 'lu'} # works 1 iteration seems same as _psi1 in time
    param_psi45 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'pc_type': 'lu'} # 1 iteration same _psi1; at A=1 (too high back at A=0.2), CFL=2 collapses but gets further than 46; 45 at A=0.2 and cfl=2 gives energy decay; 0.6m waves on 1m depth; best one? # use of ilu recommended for small time steps
    # For Laplacian part use 'gamg' or 'hypre' 
    param_psi455 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'pc_type': 'ilu'}
    param_psi4555 = {'snes_atol': 1e-14, 'ksp_converged_reason':None, "ksp_rtol": 1e-8, 'fieldsplit_0_ksp_type': 'gmres', 'fieldsplit_0_pc_type': 'ilu',\
                     'fieldsplit_1_ksp_type': 'gmres', 'fieldsplit_1_pc_type': 'ilu',\
                     'fieldsplit_2_ksp_type': 'gmres', 'fieldsplit_2_pc_type': 'gamg'}
    param_psi4555m = {'snes_atol': 1e-14, 'ksp_converged_reason':None, "ksp_rtol": 1e-8, 'fieldsplit_0_ksp_type': 'gmres', 'fieldsplit_0_pc_type': 'ilu',\
                     'fieldsplit_1_ksp_type': 'gmres', 'fieldsplit_1_pc_type': 'ilu',\
                      'fieldsplit_2_ksp_type': 'gmres', 'fieldsplit_2_pc_type': 'gamg','pc_factor_mat_solver_type':'mumps','snes_monitor':None, 'ksp_monitor':None}
    param_psi46 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'snes_type': 'vinewtonrsls', 'pc_type': 'lu'} # 1 iteration;  same as _psi1; collapsed at A=1, CFL=2 after going 1,2,3,2 iterations
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
    param_psi = {"ksp_rtol": 1e-8, 'ksp_type': 'preonly', 'pc_type': 'lu'}
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
    vpoly=10

    VP3dpf = (- H0*Ww*fd.inner(psimp, (h_new - h_old)/dt) \
              + H0*fd.inner(hmp, (Ww*psii - Ww*psi_f)/dt) \
              + H0*gg*Ww*( 0.5*fd.inner(fac*H0+hmp, fac*H0+hmp)-(fac*H0+hmp)*H0+0.5*H0**2 ) ) * fd.ds_t(vpoly) \
                + 0.5*( (Lw**2/Ww)*(fac*H0+hmp)*(u0+psimp.dx(0)*phihat+varphimp.dx(0)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(0)+x[2]*hmp.dx(0))*(psimp*dphihat+varphimp.dx(2)))**2 \
                        + Ww*(fac*H0+hmp)*(u0py+psimp.dx(1)*phihat+varphimp.dx(1)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(1)+x[2]*hmp.dx(1))*(psimp*dphihat+varphimp.dx(2)))**2 \
                        + Ww*(H0**2/(fac*H0+hmp)) * (psimp*dphihat+varphimp.dx(2))**2 ) * fd.dx(vpoly)
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
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=param_psi4555)
    
    # phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=parameters)
elif nvpcase=="SV": # Stormer-Verlet
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
                + 0.25*( (Lw**2/Ww)*(fac*H0+h_new)*(u0+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(0)+x[2]*h_new.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(fac*H0+h_new)*(psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(1)+x[2]*h_new.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(H0**2/(fac*H0+h_new)) * (psisv*dphihat+varphisv.dx(2))**2 \
                        + (Lw**2/Ww)*(fac*H0+h_old)*(u0+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(0)+x[2]*h_old.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(fac*H0+h_old)*(u0py+psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(1)+x[2]*h_old.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                        + Ww*(H0**2/(fac*H0+h_old)) * (psisv*dphihat+varphisv.dx(2))**2  ) * fd.dx

    # Step-1-2: solve psisv, varphisv variation wrt h_old (eta_old) and varphisv
    psif_exprnl1 = fd.derivative(VP3dpf, h_old, du=vvsv0) # du=v_W represents perturbation 
    phi_exprnl1 = fd.derivative(VP3dpf, varphisv, du=vvsv1)
    Fexprnl = psif_exprnl1+phi_exprnl1
    param_psi = {'ksp_type': 'preonly', 'pc_type': 'lu'} # works but inefficient? 1 iteration (just observed by trying a few tme steps)
    param_psi2 = {'ksp_converged_reason':None, 'ksp_type': 'preonly', 'pc_type': 'lu'} # works but inefficient?
    param_psi3 = {'ksp_converged_reason':None, 'ksp_type': 'cg', 'pc_type': 'hypre'} # 7 iterations (just observed by trying a few tme steps)
    param_psi1 = {'ksp_converged_reason':None, 'ksp_type': 'cg', 'pc_type': 'lu'} # 1 iteration
    param_psi4 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'pc_type': 'lu'} # 1 iteration
    param_psi44 = {'ksp_type': 'gmres', 'pc_type': 'lu'} # 1 iteration
    param_psi45 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'pc_type': 'lu'} # 1 iteration; best one? 'snes_type': 'ksponly' is linear case
    param_psi46 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'snes_type': 'vinewtonrsls', 'pc_type': 'lu'} # 1 iteration; no fine test but 4,44,45 equally fast or slow?
    param_psi5 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'pc_type': 'hypre'} # 7 iterations
    param_h       = {'ksp_converged_reason':None, 'pc_type': 'fieldsplit','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'} # used in Gidel's code 2 iterations
    param_h       = {'ksp_converged_reason':None, 'pc_type': 'fieldsplit','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'} # used in Gidel's code              
    phi_combonlsv = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedsv, bcs = BC_varphi_mixedsv), solver_parameters=param_psi46)
    #  phi_combonlsv = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedsv, bcs = BC_varphi_mixedsv), solver_parameters=param_h)
    
    #  Step-3: solve h_new=h^(n+1/2) variation wrt psi^(n+1/2)
    h_exprnl1 = fd.derivative(VP3dpf, psisv, du=v_R)
    h_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(h_exprnl1,h_new), solver_parameters=param_psi)

    #  Step-4: variation wrt h_new fsolve psii=psi^(n+1)
    psin_exprnl1 = fd.derivative(VP3dpf, h_new, du=v_R)
    phin_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(psin_exprnl1 ,psii), solver_parameters=param_psi)

# 
    

    
###### OUTPUT FILES and initial PLOTTING ##########
outfile_psi = fd.File("results/psi.pvd")
outfile_height = fd.File("results/height.pvd")
outfile_varphi = fd.File("results/varphi.pvd")
outfile_sicks = fd.File("results/sicks.pvd")
outfile_sickss = fd.File("results/sickss.pvd")

#

t = 0.0
i = 0.0

outfile_height.write(h_old, time=t)
outfile_psi.write(psi_f, time=t)
if nic=='SP2':
    outfile_sicks.write(sick, time=t)
    outfile_sickss.write(U0c0sickos, time=t)


print('Plotting starts, initial data:')
plt.figure(1)
eta1vals = np.array([h_old.at(x, yslice, zslice) for x in xvals]) # 
phi1vals = np.array([psi_f.at(x, yslice, zslice) for x in xvals])
phi1vals12 = np.array([psi_f.at(x, yslice12, zslice) for x in xvals])
phi1vals10 = np.array([psi_f.at(x, yslice10, zslice) for x in xvals])

ax1.plot(xvals, eta1vals, '-k', label = f' $\eta_n: t = {t:.3f}$',linewidth=2)
ax2.plot(xvals, phi1vals, '-k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)
ax2.plot(xvals, phi1vals12, '-k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)
ax2.plot(xvals, phi1vals10, '-k', label = f' $\phi_n: t = {t:.3f}$', linewidth=2)

if nic=="SP2":
    print('Hallocorr?')
    outfile_Y0cor = fd.File("results/Y0cor.pvd")
    outfile_Y0cor.write(U0cor, time=t)
    outfile_c0cor = fd.File("results/c0cor.pvd")
    outfile_c0cor.write(c0cor, time=t)
    U0yvals = np.array([U0cor.at(xslice, y, zslice) for y in yvals])
    U0yvals12 = np.array([U0cor.at(0.5*xslice, y, zslice) for y in yvals])
    c0yvals = np.array([c0cor.at(xslice, y, zslice) for y in yvals])
    c0yvals = np.array([c0cor.at(0.5*xslice, y, zslice) for y in yvals])
    plt.figure(33)
    plt.plot(yvals,U0yvals,'-r')
    plt.plot(yvals,U0yvals12,'-.k')
    plt.plot(yvals,c0yvals,'--b')
    plt.xlabel(f'$y$',fontsize=size)
    plt.ylabel(f'$U_0,c_0$',fontsize=size)

    
# Not yet calculated: outfile_varphi.write(varphi, time=t)


###### TIME LOOP ##########
print('Time Loop starts')
# Timer:
tic = tijd.time()
while t <= (t_end + dt): #  t_end + dt
    tt = format(t, '.3f')

    if nvpcase == "MMP": # VP MMP
        print('Hallom2 in time loop')
        phi_combonl.solve()
        print('Hallom1 in time loop')
        psimp, hmp, varphimp = result_mixedmp.split()
        psi_f.interpolate(2.0*psimp-psi_f) # update n+1 -> n
        h_old.interpolate(2.0*hmp-h_old) # update n+1 -> n
        print('Hallo0 in time loop')
        varphi.interpolate(varphimp+psi_f) # total velocity potential for plotting
        print('Hallo in time loop')
    elif nvpcase == "SV": # VP SV
        phi_combonlsv.solve()
        psisv, varphisv = result_mixedsv.split()
        h_exprnl.solve()
        phin_exprnl.solve()
        psi_f.assign(psii)
        # Done later since needed in energy EKin: h_old.assign(h_new)
        varphi.interpolate(varphisv+psi_f) # total velocity potential for plotting
        

    # Energy monitoring: bit too frequent reduce
    if t>=0.0:
        if nvpcase=="MMP":
            EKin = fd.assemble( 0.5*( (Lw**2/Ww)*(fac*H0+hmp)*(u0+psimp.dx(0)*phihat+varphimp.dx(0)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(0)+x[2]*hmp.dx(0))*(psimp*dphihat+varphimp.dx(2)))**2 \
                                      + Ww*(fac*H0+hmp)*(dU0dy*x[0]+dc0dy+psimp.dx(1)*phihat+varphimp.dx(1)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(1)+x[2]*hmp.dx(1))*(psimp*dphihat+varphimp.dx(2)))**2 \
                                      + Ww*(H0**2/(fac*H0+hmp))*(psimp*dphihat+varphimp.dx(2))**2 ) * fd.dx )
            EPot = fd.assemble( H0*gg*Ww*( 0.5*fd.inner(fac*H0+hmp,fac*H0+hmp)-(fac*H0+hmp)*H0+0.5*H0**2 ) * fd.ds_t )
        elif nvpcase=="SV":
            EKin = fd.assemble( 0.25*( (Lw**2/Ww)*(fac*H0+h_new)*(u0+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(0)+x[2]*h_new.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(fac*H0+h_new)*(dU0dy*x[0]+dc0dy+psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(1)+x[2]*h_new.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(H0**2/(fac*H0+h_new)) * (psisv*dphihat+varphisv.dx(2))**2 \
                                       + (Lw**2/Ww)*(fac*H0+h_old)*(u0+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(0)+x[2]*h_old.dx(0))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(fac*H0+h_old)*(dU0dy*x[0]+dc0dy+psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(1)+x[2]*h_old.dx(1))*(psisv*dphihat+varphisv.dx(2)))**2 \
                                       + Ww*(H0**2/(fac*H0+h_old)) * (psisv*dphihat+varphisv.dx(2))**2  ) * fd.dx  )
            EPot = fd.assemble( (0.5*H0*gg*Ww*( 0.5*fd.inner(fac*H0+h_new, fac*H0+h_new)-(fac*H0+h_new)*H0+0.5*H0**2 ) \
                                + 0.5*H0*gg*Ww*( 0.5*fd.inner(fac*H0+h_old, fac*H0+h_old)-(fac*H0+h_old)*H0+0.5*H0**2 ) ) * fd.ds_t )
            
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

        if nvpcase == "MMP" or "SV": # 
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
# End time loop
toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)
print('t=',t,'tmeet=',tmeet,'tplot',t_plot)
plt.show() 
print('*************** PROGRAM ENDS ******************')
