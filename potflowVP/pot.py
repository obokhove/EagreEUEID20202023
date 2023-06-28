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
from firedrake.petsc import PETSc
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

op2.init()
parameters["coffee"]["O2"] = False

# parameters in SI units REMIS error of polynomials
gg = 9.81  # gravitational acceleration [m/s^2]

# water domain and discretisation parameters
# nic = "linearw" # choice initial condition
nic = "SP3" # single soliton SP1
nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; "SV"=Stormer-Verlet
nphihatz = "Unity" # "Unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.
Lx = 140 # [m]
Ly = 2 # [m]
Lz = 20 # [m]
H0 = Lz # rest water depth [m]
ijnx = 280
ny = 2
nz = 4
nCG = 4 # function space order horizontal
nCGvert = 2 # function space order vertical
xx1 = 0.0
yy1 = 0.0
xx2 = Lx
yy2 = Ly
nenergyplot = 0 # Put energy data in file
nsavelocalHPC = 1 # Save files locally or on HPC in nobackup; directories specified; sets "save_path"; 0 is local
nvals = 0 #  no figure used when on HPC or when silly xvals does not work
nprintout = 0 # print out on screen or in file; 1: on screen: 0: PETSC version
     
if nic=="SP1": # soliton decaying sufficiently fast to be taken as periodi using BL expansion in z
    xtilde = 10
    muu = 0.01
    H0 = 1.0
    eps = 3.0 # 0.01, 0.1 and 0.4 work
    Lx = 2*H0/np.sqrt(muu)*xtilde # twice as long as in BokhoveKalogirou2016 since their set-up is a bit close to boundaries
    Lz = H0*1.0 # not sure about this one; no scaled z used on BL?
    H0 = Lz
    nx = 280
    ny = 2
    nz = 4
    nCG = 2 # function space order horizontal
    nCGvert = 2 # function space order vertical
    nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; SV=Stormer-Verlet
    nphihatz = "Unity" # "unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.
elif nic=="SP2": # two-soliton case made x-periodic
    eps = 0.05 # WW2022-paper setting
    muu = eps**2
    Lz= 20
    H0 = Lz
    y2hat = 20
    y1hat = -y2hat
    Y2KPE = (eps/np.sqrt(muu))*(3/np.sqrt(2))**(2/3)*y2hat
    Ly = (H0/np.sqrt(muu))*(y2hat-y1hat)
    yy2 = (H0/np.sqrt(muu))*y2hat
    yy1 = (H0/np.sqrt(muu))*y1hat
    tau0 = 0.0
    k4 = (2/9)**(1/6)/(4*np.sqrt(eps)) # np.tan(thetaa)
    Atilde = 0.5*k4**2
    
    yhatstar = -(np.sqrt(muu)/eps)*(np.sqrt(2)/3)**(2/3)*np.log(2)/k4**2

    xx2hat = k4*np.sqrt(eps)*(3/np.sqrt(2))**(1/3)*y2hat  + np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*np.log(2)/k4
    
    X1KPE = k4**2*tau0-(1.0/k4)*np.arccosh( np.exp(k4**2*Y2KPE)-2*np.exp(-k4**2*Y2KPE) )
    X2KPE = k4**2*tau0+(1.0/k4)*np.arccosh( np.exp(k4**2*Y2KPE)-2*np.exp(-k4**2*Y2KPE) )
    x2hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X2KPE+(1/eps)*np.sqrt(0.5*muu/eps)*tau0)
    x1hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X1KPE+(1/eps)*np.sqrt(0.5*muu/eps)*tau0)
    yfac = 1.0
    xx1 = yfac*(H0/np.sqrt(muu))*x1hat
    xx2 = yfac*(H0/np.sqrt(muu))*x2hat
    Lx = xx2-xx1
    value = ((2/9)**(1/6)*0.25/(0.87013))**2
    if nprintout == 1:
        print('Lx,Ly,x1hat,x2hat,y1hat,y2hat,Lz: ',Lx,Ly,x1hat,x2hat,y1hat,y2hat,Lz)
        print('xx2hat,yhatstar,k4: ',xx2hat,yhatstar,k4,value,np.log(2))
    else:
        PETSc.Sys.Print('Lx,Ly,x1hat,x2hat,y1hat,y2hat: ',Lx,Ly,x1hat,x2hat,y1hat,y2hat)
        PETSc.Sys.Print('xx2hat,yhatstar,k4: ',xx2hat,yhatstar,k4,value,np.log(2))
    nCG = 2
    # nx = int(np.ceil(132*9/25))
    # ny = int(np.ceil(480*9/25))
    nx = int(np.ceil(124*2/nCG))
    ny = int(np.ceil(480*2/nCG))
    nz = 4
     # function space order horizontal
    nCGvert = 2 # function space order vertical
    nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; SV=Stormer-Verlet
    nphihatz = "Unity" # "unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.
elif nic=="SP3": # three-soliton case made x-periodic
    eps = 0.01 # WW2022-paper setting
    muu = eps**2
    Lz = 20
    H0 = Lz
    that0 = -105 # BLE time all seems to work for that0=0 but not for that0=-200 JUNHO: please check.
    ntflag = 1
    tau0 = that0*np.sqrt(2*eps/muu)*eps # BLE to KPE time # that0*np.sqrt(0.5*muu/eps)/eps error fixed by Junho
    #
    # tantheta = (2/9)**(1/6)/(4*np.sqrt(eps)) # np.tan(thetaa) See Table 1 in WW2022 paper
    deltaa = 10**(-5)   # value of delta used in WW-paper  ; tantheta and deltaa, so eps and deltaa specify matters.
    Atilde = 0.3*(3/4)**(1/3) # Expression (26b) in WW-paper
    # tantheta = (2/9)**(1/6)/(4*np.sqrt(eps)) # np.tan(thetaa) See Table 1 in WW2022 paper
    # deltaa = 0.00140455   # value of delta used in WW-paper  ; tantheta and deltaa, so eps and deltaa specify matters.
    # Atilde = 0.5*(tantheta/(2+deltaa*np.sqrt(2)))**2
    k4 = np.sqrt(Atilde*0.5) # for lambda = 1 case  # Next 3 expressions: (25) in WW-paper
    k5 = Atilde**.5*(.5**.5+deltaa)
    k6 = Atilde**.5*((2)**.5+.5**.5+deltaa)
    k1 = -k6
    k2 = -k5
    k3 = -k4
    
    # k4 = np.sqrt(0.5*Atilde) # for lambda = 1 case  # Next 3 expressions: (25) in WW-paper
    # k5 = 0.5*tantheta-k4
    # k6 = 0.5*tantheta+k4
    # k1 = -k6
    # k2 = -k5
    # k3 = -k4
    bb = 1
    aa = (k6*(k6**2-k4**2)/(k5*(k5**2-k4**2)))**.5
    cc = 1/aa
    A235 = 2*aa*k5*(k5**2-k4**2)
    A135 = -k4*k5**2-k5*k4**2+k6*k5**2-k6*k4**2+k5*k6**2+k4*k6**2
    A136 = 2*cc*k6*(k6**2-k4**2)
    A246 = aa*bb*cc*A135
    A146 = bb*A136
    A236 = aa*cc*(-k4*k6**2-k6*k4**2+k5*k6**2-k5*k4**2+k6*k5**2+k4*k5**2) # Error fixed by Junho! 02-04-2023. Hurray. Had typed in term twice by accident.
    # A6 = cc*(k6*k6*(k3-k1) + k3*k3*(k1-k6) + k1*k1*(k6-k3))
    # A135 = k5*k5*(k3-k1) + k3*k3*(k1-k5) + k1*k1*(k5-k3)
    # A246 13= aa*bb*cc*(k6*k6*(k4-k2) + k4*k4*(k2-k6) + k2*k2*(k6-k4))
    # A146 = bb*cc*(k6*k6*(k4-k1) + k4*k4*(k1-k6) + k1*k1*(k6-k4))
    # print(A135,A246,A235,b*A35,A136,A146,A236)
    # input('sdfgf')
    Lyhat = 30 # BLE yhat-scale 
    Ystar = 0.0 # KPE Y-scale
    y1hat = Ystar*np.sqrt(muu)*(np.sqrt(2)/3)**(2/3)/eps # KPE Y-scale to BLE yhat-scale
    y2hat = y1hat+Lyhat
    Y2KPE = (eps/np.sqrt(muu))*(3/np.sqrt(2))**(2/3)*y2hat # BLE yhat to KPE Y-scale
    Ly = (H0/np.sqrt(muu))*(y2hat-y1hat) # BLE yhat to PFE y-scale
    yy2 = (H0/np.sqrt(muu))*y2hat # BLE yhat to PFE y-scale
    yy1 = (H0/np.sqrt(muu))*y1hat # BLE yhat to PFE y-scale

    # From SP2 into SP3; approximations!
    x2hat = (k5+k6)*(4.5)**(1/6)*(muu/eps)**.5*y2hat+(1+(k6**2+k6*k5+k5**2)*(4/3)**(1/3)*eps)*that0\
        - (muu/eps)**0.5*(2/9)**(1/6)*np.log(A246/A146)/(k6-k5)# KPE X-scale to BLE xhat-scale
    x1hat = -(k5+k6)*(4.5)**(1/6)*(muu/eps)**.5*y2hat+(1+(k6**2+k6*k5+k5**2)*(4/3)**(1/3)*eps)*that0\
        - (muu/eps)**0.5*(2/9)**(1/6)*np.log(A136/A135)/(k6-k5)# KPE X-scale to BLE xhat-scale
    
    # x2hat = (k5+k6)*(4.5)**(1/6)*(muu/eps)**.5*y2hat
    # x1hat =-(k5+k6)*(4.5)**(1/6)*(muu/eps)**.5*y2hat
    # x2hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X2KPE+ntflag*(1/eps)*np.sqrt(0.5*muu/eps)*tau0) # KPE X-scale to BLE xhat-scale
    # x1hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X1KPE+ntflag*(1/eps)*np.sqrt(0.5*muu/eps)*tau0) # KPE X-scale to BLE xhat-scale
    xx1 = (H0/np.sqrt(muu))*x1hat # BLE xhat to PFE x-scale
    xx2 = (H0/np.sqrt(muu))*x2hat # BLE xhat to PFE x-scale
    Lx = xx2-xx1
    value = ((2/9)**(1/6)*0.25/(0.87013))**2
    if nprintout == 1:
        print('Lx,Ly,x1hat,x2hat,y1hat,y2hat: ',Lx,Ly,x1hat,x2hat,y1hat,y2hat) # printing to compare wth WW-paper data
        print('Ystar,k4,log(2.7),xx1,xx2: ',Ystar,k4,value,np.log(2.7),xx1,xx2) # printing to compare wth WW-paper data
    else:
        PETSc.Sys.Print('Lx,Ly,x1hat,x2hat,y1hat,y2hat: ',Lx,Ly,x1hat,x2hat,y1hat,y2hat) # printing to compare wth WW-paper data
        PETSc.Sys.Print('Ystar,k4,log(2.7),xx1,xx2: ',Ystar,k4,value,np.log(2.7),xx1,xx2) # printing to compare wth WW-paper data
    nCG = 2 # function space order horizontal
    nCGvert = 2 # function space order vertical
    cxx=5*4/nCG
    # else: c=12
    nx = int(cxx*np.ceil(x2hat-x1hat))
    ny = int(cxx*np.ceil(Lyhat))
    nz = 4
    
    nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; SV=Stormer-Verlet
    nphihatz = "Unity" # "unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.                                      

#__________________  FIGURE PARAMETERS  _____________________#


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
                               quadrilateral=True, reorder=None,distribution_parameters=None,diagonal=None,comm=COMM_WORLD,name='mesh2d') # xx1, xx2, yy1, yy2
mesh = fd.ExtrudedMesh(mesh2d, nz, layer_height=Lz/nz, extrusion_type='uniform',name='mesh')
# x, y, z = fd.SpatialCoordinate(mesh)
x = mesh.coordinates
top_id = 'top'
# 

# For in-situ plotting



# print('Lx,Ly,Lz,xslice,yslice,zslice',Lx,Ly,Lz,H0,xslice,yslice,zslice)
# print('xyz',mesh.coordinates.dat.data_min)
# print('mesh issues:',np.minimum(fd.interpolate(x[0])),np.maximum(fd.interpolate(x[0])))

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
    if nprintout==1:
        print('Period: ', Tperiod)
    else:
        PETSc.Sys.Print('Period: ', Tperiod)

    psi_exact_expr = D * fd.sin(kx * x[0]-omega * t0) * fd.cosh(kx * x[2])
    psi_exact_exprH0 = D * fd.sin(kx * x[0]-omega * t0) * fd.cosh(kx * H0)
    eta_exact_expr = A * fd.cos(kx * x[0]-omega * t0)
    btopoexpr = 0.0*psi_exact_exprH0
    dtt = np.minimum(Lx/nx,Lz/nz)/(np.pi*np.sqrt(gg*H0)) # i.e. dx/max(c0) with c0 =sqrt(g*H0)
    Nt = 500 # check with print statement below and adjust dt towards dtt vi Nt halving time step seems to half energy oscillations
    CFL = 1.0 # run at a) 0.125 and b) 0.5*0.125 2,0 for mmp! CFL=0.5 for SV time-step restriction!
    CFL = 0.5
    dt = CFL*Tperiod/Nt  # 0.005  # time step [s]
    nTfac = 2
    t_end = nTfac*Tperiod  # time of simulation [s]
    if nprintout==1:
        print('dtt=',dtt, t_end/dtt,dt,2/omega)
    else:
        PETSc.Sys.Print('dtt=',dtt, t_end/dtt,dt,2/omega)
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    
    nplot = 16*nTfac
elif nic=="SP1": # SP1 soliton periodic dimensional
    tau0=1
    t0 = (H0/np.sqrt(gg*H0*muu))*tau0
    cc = 0.5
    x0 = 0.5*Lx # Halfway again as in BokhoveKalogirou2016
    qq = 0.5*np.sqrt(cc*eps/muu)*((np.sqrt(muu)/H0)*(x[0]-x0)-((1+eps*cc/6))*(np.sqrt(muu*gg*H0)/H0)*t0) # argument used in soliton dimensonal
    eta_exact_expr = eps*H0*(cc/3.0)*fd.cosh(qq)**(-2)
    Atilde = eps*H0*(cc/3.0)
    xx0 = 0
    qq0 = 0.5*np.sqrt(cc*eps/muu)*((np.sqrt(muu)/H0)*(xx0-x0)-((1+eps*cc/6))*(np.sqrt(muu*gg*H0)/H0)*t0) # argument used in soliton dimensonal at 0
    qqLx = 0.5*np.sqrt(cc*eps/muu)*((np.sqrt(muu)/H0)*(Lx-x0)-((1+eps*cc/6))*(np.sqrt(muu*gg*H0)/H0)*t0) # argument used in soliton dimensonal at Lx
    psiqq0_exact_exprH0 = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qq0) \
                                            +0.25*(eps*muu/H0**2)*((H0+eta_exact_expr)**2)*fd.sinh(qq0)/fd.cosh(qq0)**3 )
    psiLx_exact_exprH0 = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qqLx) \
                                            +0.25*(eps*muu/H0**2)*((H0+eta_exact_expr)**2)*fd.sinh(qqLx)/fd.cosh(qqLx)**3 )
    u0 = (psiLx_exact_exprH0-psiqq0_exact_exprH0)/Lx + 0.0*x[1]
    U0yxc0y = u0*x[0] 
    u0py =  u0.dx(1)
    psi_exact_exprH0 = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qq) \
                                            +0.25*(eps*muu/H0**2)*((H0+eta_exact_expr)**2)*fd.sinh(qq)/fd.cosh(qq)**3 ) - u0*x[0] # that's it I think?
    
    psiqq0exactz = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qq0)+0.25*(eps*muu/H0**2)*(x[2]**2)*fd.sinh(qq0)/fd.cosh(qq0)**3 )
    psiLxexactz = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qqLx)+0.25*(eps*muu/H0**2)*((x[2])**2)*fd.sinh(qqLx)/fd.cosh(qqLx)**3 )
    u0z = (psiLxexactz-psiqq0exactz)/Lx + 0.0*x[1]
    c0z = Lx*psiqq0exactz/Lx
    U0yxc0yz = c0z+u0z*x[0]+0.0*x[1]
    u0zpy = U0yxc0yz.dx(1)
    u0zpz = U0yxc0yz.dx(2)
    c0zpy = c0z.dx(1)
    c0zpz = c0z.dx(2)
    psi_exactz = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qq) \
                                            +0.25*(eps*muu/H0**2)*(x[2]**2)*fd.sinh(qq)/fd.cosh(qq)**3 ) - u0z*x[0] # that's it I think?
    
    psi_exact_exprH0 = (eps*H0*np.sqrt(gg*H0/muu))*(2/3)*np.sqrt(cc*muu/eps)*( 1.0+fd.tanh(qq) \
                                            +0.25*(eps*muu/H0**2)*((H0+eta_exact_expr)**2)*fd.sinh(qq)/fd.cosh(qq)**3 ) - u0*x[0] # that's it I think?
    
    btopoexpr = 0.0*psi_exact_exprH0 # no topography
    ttilde = 13 # BLE time
    t_end = ttilde*(H0/np.sqrt(gg*H0*muu)) # dimensionless BLE end-time of BokhoveKalogirou2016 times time-scaling factor
    Nt = tau0*100
    CFL = 0.25
    dt = t_end/Nt # Somehow depends on soliton speed; so dx/U0 with U0=x/t dimensional from argument soliton; see dtt
    dtt = CFL*(Lx/nx)*6.0/((1+eps*cc)*np.sqrt(gg*H0))
    dt = dtt
    if nprintout==1:
        print('dtt=',dt,dtt,t_end/dtt)
        print('t_end: ',t_end)
    else:
        PETSc.Sys.Print('dtt=',dt,dtt,t_end/dtt)
        PETSc.Sys.Print('t_end: ',t_end)
        
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 10

    # to do
elif nic=="SP2": # SP2 solitons periodic dimensional; extend VP? Define eta and phi
    t0 = 0.0
    # xx1, xx2, yy2=Ly
    Fx = np.sqrt(eps/muu)*(3/np.sqrt(2))**(1/3)
    Fy = (eps/np.sqrt(muu))*(3/np.sqrt(2))**(2/3)
    
    # xx1 used x[1]=y function ; can this be automated via general case? x[0]-0.5*Lx , x[1]-0.5*Ly ; 0...Lx=(x2-x1)
    xx11 = 0.0 # 0.0
    xx22 = Lx # Lx
    xs = -xx1
    ys = -yy1
    if nprintout==1:
        print('xx1, xx2, yy1, yy2', xx1, xx2, yy1, yy2)
    else:
        PETSc.Sys.Print('xx1, xx2, yy1, yy2', xx1, xx2, yy1, yy2)    
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
    etax1y = 2.0*eps*H0*((4/3)**(1/3))*( KKXX1/KK1 -(KKX1/KK1)**2 )
    psix1y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1-0.5*muu*((H0+0.0*etax1y)/H0)**2*(GX1+GY1))
    psix1yz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1-0.5*muu*((x[2])/H0)**2*(GX1+GY1))
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
    psix2y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2-0.5*muu*((H0+0.0*etax2y)/H0)**2*(GX2+GY2))
    psix2yz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2-0.5*muu*((x[2])/H0)**2*(GX2+GY2))
    # psix2y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2)

    U0y = (psix2y-psix1y)/(xx2-xx1)
    c0y = (xx2*psix1y-xx1*psix2y)/(xx2-xx1)
    U0yxc0y = U0y*x[0]+c0y # (U0y*x[0]+c0y)
    sicko = U0y*x[0]+c0y
    u0 = U0yxc0y.dx(0)
    u0py = U0yxc0y.dx(1)*x[0]
    
    u0z = (psix2yz-psix1yz)/(xx2-xx1)
    c0z = (xx2*psix1yz-xx1*psix2yz)/(xx2-xx1)
    # U0yxc0yz = u0z*x[0]+0.0*x[1]
    U0yxc0yz = c0z+u0z*x[0]+0.0*x[1]
    u0zpy = U0yxc0yz.dx(1)
    u0zpz = U0yxc0yz.dx(2)

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
    eta_exact_expr = 2.0*eps*H0*((4/3)**(1/3))*( KKXX/KK -(KKX/KK)**2 )
    psi_exact_exprH0 = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu*((H0+eta_exact_expr)/H0)**2*(GX+GY))-(u0*x[0]+c0y)
    psi_exact_exprz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu*((x[2])/H0)**2*(GX+GY))-(u0z*x[0]+c0z)
    sickofit =         ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu*((H0+eta_exact_expr)/H0)**2*(GX+GY))-(u0*x[0]+c0y)
    
    btopoexpr = 0.0*psi_exact_exprH0 # no topography
    ttilde = 60 # time units used for SP2 in BLE
    # ttilde = 2
    t_end = ttilde*(H0/np.sqrt(gg*H0*muu)) # dimensionless BLE end-time of BokhoveKalogirou2016 times time-scaling factor
    # t2hat = eps*np.sqrt(2*eps/mu**2) # BLE to PFE time scaling
    # t_end = ttilde*t2hat
    Nt = ttilde*200
    # Nt = 80*10
    dt = t_end/Nt # Somehow depends on soliton speed; so dx/U0 with U0=x/t dimensional from argument soliton; see dtt
    CFL = 0.5
    cc = 1
    dtt = CFL*(Lx/nx)*6.0/((1+eps*cc)*np.sqrt(gg*H0))    
    
    nplot = 10
    tijd.sleep(0.1)
    
elif nic=="SP3": # SP3 solitons periodic dimensional; extend VP?
    ttilde = 50 # BLE final time units used for SP3 in BLE
    t_end = ttilde*(H0/np.sqrt(gg*H0*muu)) # dimensionless BLE end-time of WW-paper or otherwise times time-scaling factor
    t0 = that0*(H0/np.sqrt(gg*H0*muu)) 
    # xx1, xx2, yy2=Ly
    Fx = np.sqrt(eps/muu)*(3/np.sqrt(2))**(1/3)
    Fy = (eps/np.sqrt(muu))*(3/np.sqrt(2))**(2/3)
    
    # xx1 used x[1]=y function ; can this be automated via general case? x[0]-0.5*Lx , x[1]-0.5*Ly ; 0...Lx=(x2-x1)
    xx11 = 0.0 # 0.0
    xx22 = Lx # Lx
    ntflag2 = 0 # Junho's shift iff ntflag2 = ntflag = 0 illegal or not? How can it be justfied? 31032023 02-04-2023: ntflag2 =1 does not seem to work
    fac0 = 1 # factor 0 or 1 to switch off or on the O(mu^2) terms
    xs = -xx1-ntflag2*np.sqrt(gg*H0)*t0 # xs+ntflag*np.sqrt(gg*H0)*t0 = -xx1 when ntflag2=ntflag
    ys = -yy1
    if nprintout==1:
        print('xx1, xx2, yy1, yy2', xx1, xx2, yy1, yy2)
    else:
        PETSc.Sys.Print('xx1, xx2, yy1, yy2', xx1, xx2, yy1, yy2)
        
    alphaa = -np.log(np.sqrt(aa*cc))
    XX1 = (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3)*(xx11-xs-ntflag*np.sqrt(gg*H0)*t0) # sqrt(eps/mu)*(3/sqrt(2))**(1/3)
    Y2 = (eps/H0)*(3/np.sqrt(2))**(2/3)*(x[1]-ys) # PFE to BLE to KPE Y-scale
    KK1 = 2*fd.exp((k4**2+k5**2+k6**2)*Y2)*( A135*np.exp(-alphaa)*np.cosh(-3*k4*XX1+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX1-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.cosh(k4*XX1+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKX1 = 2*k4*fd.exp((k4**2+k5**2+k6**2)*Y2)*(-3*A135*np.exp(-alphaa)*np.sinh(-3*k4*XX1+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX1-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.sinh(k4*XX1+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKXX1 = 2*k4**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*( 9*A135*np.exp(-alphaa)*np.cosh(-3*k4*XX1+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX1-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.cosh(k4*XX1+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKXXX1 =  2*k4**3*fd.exp((k4**2+k5**2+k6**2)*Y2)*(-27*A135*np.exp(-alphaa)*np.sinh(-3*k4*XX1+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX1-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.sinh(k4*XX1+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKY1 = (k4**2+k5**2+k6**2)*KK1 \
        +4*A136*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX1-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar))
    KKYY1 = (k4**2+k5**2+k6**2)**2*KKY1 \
        +4*A136*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX1-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar)) \
        +4*A136*(k6**2-k5**2)**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX1-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar))
    KKXY1 = (k4**2+k5**2+k6**2)*KKX1 \
        +4*k4*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX1-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar))
    KKXYY1 = (k4**2+k5**2+k6**2)**2*KKXY1 \
        +4*k4*A136*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX1-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar)) \
        +4*k4*A136*(k6**2-k5**2)**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX1-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar))
    GX1 = 2*Fx**2*( KKXXX1/KK1-3*KKXX1*KKX1/KK1**2+2*KKX1**3/KK1**3 )
    GY1 = 2*Fy**2*( KKXYY1/KK1-2*KKXY1*KKY1/KK1**2-KKX1*KKYY1/KK1**2+2*KKX1*(KKY1)**2/KK1**3 )
    etax1y = 2.0*eps*H0*((4/3)**(1/3))*( KKXX1/KK1 -(KKX1/KK1)**2 )
    psix1y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1-fac0*0.5*muu*((H0+0.0*etax1y)/H0)**2*(GX1+GY1))
    psix1yz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1-fac0*0.5*muu*((x[2])/H0)**2*(GX1+GY1))

    # xx2 used x[1]=y function ; can this be automated via general case?
    XX2 = (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3)*(xx22-xs-ntflag*np.sqrt(gg*H0)*t0)
    KK2 = 2*fd.exp((k4**2+k5**2+k6**2)*Y2)*( A135*np.exp(-alphaa)*np.cosh(-3*k4*XX2+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX2-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.cosh(k4*XX2+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKX2 = 2*k4*fd.exp((k4**2+k5**2+k6**2)*Y2)*(-3*A135*np.exp(-alphaa)*np.sinh(-3*k4*XX2+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX2-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.sinh(k4*XX2+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKXX2 = 2*k4**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*( 9*A135*np.exp(-alphaa)*np.cosh(-3*k4*XX2+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX2-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.cosh(k4*XX2+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKXXX2 = 2*k4**3*fd.exp((k4**2+k5**2+k6**2)*Y2)*(-27*A135*np.exp(-alphaa)*np.sinh(-3*k4*XX2+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX2-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*np.sinh(k4*XX2+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKY2 = (k4**2+k5**2+k6**2)*KK2 \
        +4*A136*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX2-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2 -Ystar))
    KKYY2 = (k4**2+k5**2+k6**2)**2*KKY2 \
        +4*A136*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX2-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar)) \
        +4*A136*(k6**2-k5**2)**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.cosh(k4*XX2-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar))
    KKXY2 = (k4**2+k5**2+k6**2)*KKX2 \
        +4*k4*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX2-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar))
    KKXYY2 = (k4**2+k5**2+k6**2)**2*KKXY2 \
        +4*k4*A136*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX2-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar)) \
        +4*k4*A136*(k6**2-k5**2)**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*np.sinh(k4*XX2-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar))
    GX2 = 2*Fx**2*( KKXXX2/KK2-3*KKXX2*KKX2/KK2**2+2*KKX2**3/KK2**3 )
    GY2 = 2*Fy**2*( KKXYY2/KK2-2*KKXY2*KKY2/KK2**2-KKX2*KKYY2/KK2**2+2*KKX2*(KKY2)**2/KK2**3 )
    etax2y = 2.0*eps*H0*((4/3)**(1/3))*( KKXX2/KK2 -(KKX2/KK2)**2 )
    psix2y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2-fac0*0.5*muu*((H0+0.0*etax2y)/H0)**2*(GX2+GY2))
    psix2yz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2-fac0*0.5*muu*((x[2])/H0)**2*(GX2+GY2))


    U0y = (psix2y-psix1y)/(xx2-xx1)
    c0y = (xx2*psix1y-xx1*psix2y)/(xx2-xx1)
    U0yxc0y = U0y*x[0]+c0y # (U0y*x[0]+c0y)
    sicko = U0y*x[0]+c0y
    u0 = U0yxc0y.dx(0)
    u0py = U0yxc0y.dx(1)*x[0]
    
    u0z = (psix2yz-psix1yz)/(xx2-xx1)
    U0yxc0yz = u0z*x[0]+0.0*x[1]
    u0zpy = U0yxc0yz.dx(1)
    u0zpz = U0yxc0yz.dx(2)

    XX = (np.sqrt(eps)/H0)*(3/np.sqrt(2))**(1/3)*(x[0]-xs-ntflag*np.sqrt(gg*H0)*t0)
    KK = 2*fd.exp((k4**2+k5**2+k6**2)*Y2)*( A135*np.exp(-alphaa)*fd.cosh(-3*k4*XX+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*fd.cosh(k4*XX-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*fd.cosh(k4*XX+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKX = 2*k4*fd.exp((k4**2+k5**2+k6**2)*Y2)*(-3*A135*np.exp(-alphaa)*fd.sinh(-3*k4*XX+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*fd.sinh(k4*XX-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*fd.sinh(k4*XX+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKXX = 2*k4**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*( 9*A135*np.exp(-alphaa)*fd.cosh(-3*k4*XX+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*fd.cosh(k4*XX-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*fd.cosh(k4*XX+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKXXX =  2*k4**3*fd.exp((k4**2+k5**2+k6**2)*Y2)*(-27*A135*np.exp(-alphaa)*fd.sinh(-3*k4*XX+(k4**3+k6**3-k5**3)*tau0+alphaa) \
                                +2*A136*np.exp((k6**2-k5**2)*Ystar)*fd.sinh(k4*XX-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar)) \
                                +A236*np.exp(alphaa)*fd.sinh(k4*XX+(k4**3+k5**3-k6**3)*tau0-alphaa) )
    KKY = (k4**2+k5**2+k6**2)*KK \
        +4*A136*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*fd.cosh(k4*XX-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2 -Ystar))
    KKYY = (k4**2+k5**2+k6**2)**2*KKY \
        +4*A136*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*fd.cosh(k4*XX-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar)) \
        +4*A136*(k6**2-k5**2)**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*fd.cosh(k4*XX-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar))
    KKXY = (k4**2+k5**2+k6**2)*KKX \
        +4*k4*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*fd.sinh(k4*XX-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar))
    KKXYY = (k4**2+k5**2+k6**2)**2*KKXY \
        +4*k4*A136*(k4**2+k5**2+k6**2)*(k6**2-k5**2)*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*fd.sinh(k4*XX-k4**3*tau0)*fd.sinh((k6**2-k5**2)*(Y2-Ystar)) \
        +4*k4*A136*(k6**2-k5**2)**2*fd.exp((k4**2+k5**2+k6**2)*Y2)*np.exp((k6**2-k5**2)*Ystar)*fd.sinh(k4*XX-k4**3*tau0)*fd.cosh((k6**2-k5**2)*(Y2-Ystar))
    GX = 2*Fx**2*( KKXXX/KK-3*KKXX*KKX/KK**2+2*KKX**3/KK**3 )
    GY = 2*Fy**2*( KKXYY/KK-2*KKXY*KKY/KK**2-KKX*KKYY/KK**2+2*KKX*(KKY)**2/KK**3 )
    eta_exact_expr = 2.0*eps*H0*((4/3)**(1/3))*( KKXX/KK -(KKX/KK)**2 )
    psi_exact_exprH0 = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-fac0*0.5*muu*((H0+0.0*etax2y)/H0)**2*(GX+GY))-(u0*x[0])
    psi_exact_exprz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-fac0*0.5*muu*((x[2])/H0)**2*(GX+GY))-(u0z*x[0])
    sickofit =  psi_exact_exprH0 
    
    btopoexpr = 0.0*psi_exact_exprH0 # no topography
    
    Nt = ttilde*200
    dt = t_end/Nt # Somehow depends on soliton speed; so dx/U0 with U0=x/t dimensional from argument soliton; see dtt
    CFL = 0.5
    cc = 1
    dtt = CFL*(Lx/nx)*6.0/((1+eps*cc)*np.sqrt(gg*H0))    
    if nprintout==1:
        print('dtt=',dt,dtt,t_end/dtt)
    else:
        PETSc.Sys.Print('dtt=',dt,dtt,t_end/dtt)
        
    ##______________  To get results at different time steps ______________##
    while (t <= t_end+dt):
        time.append(t)
        t+= dt
    nplot = 5
    tijd.sleep(0.1)
# End if

Nenergy = 1000



##_________________  FIGURE SETTINGS __________________________##
                       


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
# err = fd.Function(V_R, name="err") # total velocity potential
# U0cor = fd.Function(V_R, name="U0cor") # 
# c0cor = fd.Function(V_R, name="c0cor") # 
# EKin0 = fd.Function(V_W, name="EKin") # 
# EPot0 = fd.Function(V_W, name="EPot") # 



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
    h_old.interpolate(eta_exact_expr)
    psi_f.interpolate(psi_exact_exprH0)
    btopo.interpolate(btopoexpr)
    varphi.interpolate(psi_exact_exprz)
    BC_varphi_mixedmp = fd.DirichletBC(mixed_Vmp.sub(2), 0, top_id) # varphimp condition for modified midpoint
elif nvpcase=="SV":
    h_old.interpolate(H0*(fac-1.0)+eta_exact_expr)
    psi_f.interpolate(psi_exact_exprH0)
    btopo.interpolate(btopoexpr)    
    BC_varphi_mixedsv = fd.DirichletBC(mixed_Vsv.sub(1), 0, top_id) # varphisv condition for modified midpoint

# if nic=="SP2" or nic=="SP3":
#     U0cor.interpolate(U0y)
#     c0cor.interpolate(c0y)
#     sick.interpolate(psi_exact_exprH0)
#     U0c0sickos.interpolate(sicko)
#     # sick.interpolate(sickofit)


if nvpcase=="MMP":

    lines_parameters = {'ksp_type': 'gmres', 'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC', 'star_construct_dim': 2,
                        'star_sub_sub_pc_type': 'lu', 'sub_sub_pc_factor_mat_ordering_type': 'rcm'}
    Ww = Lx
    Lw = Lx
    LwdWw2 = (Lw/Ww)**2
    if nphihatz=="GLL1":
        fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), nCGvert+1) # GLL
        zk = (H0*fiat_rule.get_points())
        phihat = fd.product( (x[2]-zk.item(kk))/(H0-zk.item(kk)) for kk in range(0,nCGvert-1,1) )
        dphihat = phihat.dx(2) # May not work and in that case specify the entire product: dpsidxi3 = psimp*phihat.dx(1)
    elif nphihatz=="Unity":
        phihat = 1.0
        dphihat = 0.0
    vpoly=10 # 5 seems fine too
    
    nfullgrav = 1
    if nfullgrav == 0:
        VP3dpf = (- H0*fd.inner(psimp, (h_new - h_old)/dt) \
                  + H0*fd.inner(hmp, (psii - psi_f)/dt) \
                  + H0*gg*( 0.5*fd.inner(fac*H0+hmp, fac*H0+hmp)-(fac*H0+hmp)*H0+0.5*H0**2 ) ) * fd.ds_t(degree=vpoly) \
                  + 0.5*( LwdWw2*(fac*H0+hmp)*(u0z + psimp.dx(0)*phihat+varphimp.dx(0)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(0)+x[2]*hmp.dx(0))*(u0zpz + psimp*dphihat+varphimp.dx(2)))**2 \
                          + (fac*H0+hmp)*(u0zpy + psimp.dx(1)*phihat+varphimp.dx(1)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(1)+x[2]*hmp.dx(1))*(u0zpz + psimp*dphihat+varphimp.dx(2)))**2 \
                          + (H0**2/(fac*H0+hmp)) * (u0zpz + psimp*dphihat+varphimp.dx(2))**2 ) * fd.dx(degree=vpoly) 
        #   - H0*gg*Ww*( 0.5*fd.inner(fac*H0+hmp, fac*H0+hmp)-(fac*H0+hmp)*H0+0.5*H0**2 ) * fd.ds_b(degree=vpoly)
    else:
        VP3dpf = (- H0*fd.inner(psimp, (h_new - h_old)/dt) \
                  + H0*fd.inner(hmp, (psii - psi_f)/dt) ) * fd.ds_t(degree=vpoly) \
                  + ( 0.5*( LwdWw2*(fac*H0+hmp)*(u0z + psimp.dx(0)*phihat+varphimp.dx(0)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(0)+x[2]*hmp.dx(0))*(u0zpz + psimp*dphihat+varphimp.dx(2)))**2 \
                          + (fac*H0+hmp)*(u0zpy + psimp.dx(1)*phihat+varphimp.dx(1)-(1.0/(fac*H0+hmp))*(H0*btopo.dx(1)+x[2]*hmp.dx(1))*(u0zpz + psimp*dphihat+varphimp.dx(2)))**2 \
                            + (H0**2/(fac*H0+hmp)) * (u0zpz + psimp*dphihat+varphimp.dx(2))**2 ) \
                     + gg*H0*(fac+hmp/H0)*(x[2]*(fac*H0+hmp)/H0-H0) ) * fd.dx(degree=vpoly)
                      #+ gg*(hmp+H0)*(0.5*(H0+hmp)-H0) ) * fd.dx(degree=vpoly)
    
    # Step-1: solve h^(n+1/2) wrt psi^(n+1/2)
    psif_exprnl1 = fd.derivative(VP3dpf, psimp, du=vvmp0) # du=v_W represents perturbation 
    psif_exprnl1 = fd.replace(psif_exprnl1, {psii: 2.0*psimp-psi_f})
    psif_exprnl1 = fd.replace(psif_exprnl1, {h_new: 2.0*hmp-h_old}) 
    
    # Step-2: solve psi^(n+1/2) wrt hmp=h^(n+1/2)
    h_exprnl1 = fd.derivative(VP3dpf, hmp, du=vvmp1)
    h_exprnl1 = fd.replace(h_exprnl1, {psii: 2.0*psimp-psi_f})
    h_exprnl1 = fd.replace(h_exprnl1, {h_new: 2.0*hmp-h_old})
    
    # Step-3: wrt varmp=varphi^(n+1/2) solve varmp=varphi^(n+1/2)
    phi_exprnl1 = fd.derivative(VP3dpf, varphimp, du=vvmp2)
    phi_exprnl1 = fd.replace(phi_exprnl1, {psii: 2.0*psimp-psi_f})
    phi_exprnl1 = fd.replace(phi_exprnl1, {h_new: 2.0*hmp-h_old}) 
    
    Fexprnl = psif_exprnl1+h_exprnl1+phi_exprnl1
    phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=lines_parameters)
    # phi_combonl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedmp, bcs = BC_varphi_mixedmp), solver_parameters=param_psi4555cg)

elif nvpcase=="SV": # Stormer-Verlet
    Ww = Lx
    Lw = Lx
    LwdWw2 = (Lw/Ww)**2
    if nphihatz=="GLL1":  # Correction 23-05-2023 from 2D waveflap case Wajiha
        fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), nCGvert+1) # GLL
        zk = (H0*fiat_rule.get_points())
        phihat = fd.product( (x[2]-zk.item(kk))/(H0-zk.item(kk)) for kk in range(0,nCGvert-1,1) )
        dphihat = phihat.dx(2) # May not work and in that case specify the entire product: dpsidxi3 = psimp*phihat.dx(2)
    elif nphihatz=="Unity":
        phihat = 1.0
        dphihat = 0.0

    # Variables [psisv, varphisv], h_new, psii
    # VP3dpf = (- H0*fd.inner(psisv, (h_new - h_old)/dt) + H0*fd.inner(psii, h_new/dt) - H0*fd.inner(psi_f, h_old/dt) \
    #           + 0.5*H0*gg*( 0.5*fd.inner(fac*H0+h_new, fac*H0+h_new)-(fac*H0+h_new)*H0+0.5*H0**2 ) \
    #           + 0.5*H0*gg*( 0.5*fd.inner(fac*H0+h_old, fac*H0+h_old)-(fac*H0+h_old)*H0+0.5*H0**2 )  ) * fd.ds_t \
    #             + 0.25*( LwdWw2*(fac*H0+h_new)*(u0z+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(0)+x[2]*h_new.dx(0))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
    #                     + (fac*H0+h_new)*(u0zpy + psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(1)+x[2]*h_new.dx(1))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
    #                     + (H0**2/(fac*H0+h_new)) * (u0zpz+psisv*dphihat+varphisv.dx(2))**2 \
    #                     + LwdWw2*(fac*H0+h_old)*(u0z+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(0)+x[2]*h_old.dx(0))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
    #                     + (fac*H0+h_old)*(u0zpy+psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(1)+x[2]*h_old.dx(1))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
    #                     + (H0**2/(fac*H0+h_old)) * (u0zpz +psisv*dphihat+varphisv.dx(2))**2  ) * fd.dx
    VP3dpf = (- H0*fd.inner(psisv, (h_new - h_old)/dt) + H0*fd.inner(psii, h_new/dt) - H0*fd.inner(psi_f, h_old/dt) ) * fd.ds_t \
                + 0.25*( LwdWw2*(fac*H0+h_new)*(u0z+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(0)+x[2]*h_new.dx(0))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
                        + (fac*H0+h_new)*(u0zpy + psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(1)+x[2]*h_new.dx(1))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
                        + (H0**2/(fac*H0+h_new)) * (u0zpz+psisv*dphihat+varphisv.dx(2))**2 \
                        + LwdWw2*(fac*H0+h_old)*(u0z+psisv.dx(0)*phihat+varphisv.dx(0)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(0)+x[2]*h_old.dx(0))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
                        + (fac*H0+h_old)*(u0zpy+psisv.dx(1)*phihat+varphisv.dx(1)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(1)+x[2]*h_old.dx(1))*(u0zpz + psisv*dphihat+varphisv.dx(2)))**2 \
                        + (H0**2/(fac*H0+h_old)) * (u0zpz +psisv*dphihat+varphisv.dx(2))**2\
                           + 2*gg*H0*(fac+h_old/H0)*(x[2]*(fac*H0+h_old)/H0-H0)+ 2*gg*H0*(fac+h_new/H0)*(x[2]*(fac*H0+h_new)/H0-H0) ) * fd.dx

    # Step-1-2: solve psisv, varphisv variation wrt h_old (eta_old) and varphisv
    psif_exprnl1 = fd.derivative(VP3dpf, h_old, du=vvsv0) # du=v_W represents perturbation  'snes_type': 'ksponly' linear
    phi_exprnl1 = fd.derivative(VP3dpf, varphisv, du=vvsv1)
    Fexprnl = psif_exprnl1+phi_exprnl1
    param_psi = {'ksp_type': 'cg', 'pc_type': 'none'} # works but inefficient? 1 iteration (just observed by trying a few tme steps)
    param_psi4 = {'ksp_type': 'gmres', 'pc_type': 'ilu'} # 1 iteration
    param_psi44 = {'ksp_type': 'gmres', 'pc_type': 'lu'} # 1 iteration
    param_psi45 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'pc_type': 'lu'} # 1 iteration; best one? 'snes_type': 'ksponly' is linear case
    param_psi46 = {'ksp_converged_reason':None, 'ksp_type': 'gmres', 'snes_type': 'gamg', 'pc_type': 'ilu'} # 1 iteration; no fine test but 4,44,45 equally fast or slow?
    param_h       = {'ksp_converged_reason':None, 'pc_type': 'fieldsplit','pc_fieldsplit_type': 'schur','pc_fieldsplit_schur_fact_type': 'upper'} # used in Gidel's code
    lines_parameters2 = {'ksp_type': 'gmres', 
                          'pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC',
                          'snes_lag_preconditioner': 5,
                          'star_construct_dim': 2, 'star_sub_sub_pc_type': 'lu', "star_sub_sub_pc_factor_mat_ordering_type": "rcm"}
    # lines_parameters2 = {'ksp_type': 'gmres','pc_type': 'python', 'pc_python_type': 'firedrake.ASMStarPC', 'star_construct_dim': 1,
    #                     'star_sub_sub_pc_type': 'lu'}
    # phi_combonlsv = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedsv, bcs = BC_varphi_mixedsv), solver_parameters=param_psi46)
    phi_combonlsv = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(Fexprnl, result_mixedsv, bcs = BC_varphi_mixedsv), solver_parameters=lines_parameters2)
    
    #  Step-3: solve h_new=h^(n+1/2) variation wrt psi^(n+1/2)
    h_exprnl1 = fd.derivative(VP3dpf, psisv, du=v_R)
    h_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(h_exprnl1,h_new), solver_parameters=lines_parameters2)

    #  Step-4: variation wrt h_new fsolve psii=psi^(n+1)
    psin_exprnl1 = fd.derivative(VP3dpf, h_new, du=v_R)
    phin_exprnl = fd.NonlinearVariationalSolver(fd.NonlinearVariationalProblem(psin_exprnl1 ,psii), solver_parameters=param_psi)
# 
    

###### OUTPUT FILES and initial PLOTTING ##########
if nsavelocalHPC==1: # save_path =  "/nobackup/$USER/lin_pot_flow/"
    save_path =  "/nobackup/chlw/amtjch/PFE_sp3_ep01_A03/"
else:
    save_path =  "/Users/onnobokhove/amtob/werk/vuurdraak2021/blexact/"

# if not os.path.exists(save_path):
#     os.makedirs(save_path) 

if nic=="SP3" and ntflag==0 and ntflag2==0:
    outfile_psi = fd.File(save_path+"resultsntflag0/psi.pvd")
    outfile_height = fd.File(save_path+"resultsntflag0/height.pvd")
    outfile_varphi = fd.File(save_path+"resultsntflag0/varphi.pvd")
    outfile_sicks = fd.File(save_path+"resultsntflag0/sicks.pvd")
    outfile_sickss = fd.File(save_path+"resultsntflag0/sickss.pvd")
    fileE = save_path+"resultsntflag0/potflow3dperenergy.txt"
    filEtaMax = save_path+"resultsntflag0/potflow3dperetamax.txt"
    if nprintout==1:
        print('ntflag=, ntflag2=',ntflag,ntflag2)
    else:
        PETSc.Sys.Print('ntflag=, ntflag2=',ntflag,ntflag2)
elif nic=="SP3" and ntflag==1 and ntflag2==0:
    outfile_psi = fd.File(save_path+"results1/psi.pvd")
    outfile_height = fd.File(save_path+"results1/height.pvd")
    outfile_varphi = fd.File(save_path+"results1/varphi.pvd")
    # outfile_sicks = fd.File(save_path+"results1/sicks.pvd")
    # outfile_sickss = fd.File(save_path+"results1/sickss.pvd")
    fileE = save_path+"results1/potflow3dperenergy.txt"
    filEtaMax = save_path+"results1/potflow3dperetamax.txt"
    
else:
    outfile_psi = fd.File(save_path+"results2/psi.pvd")
    outfile_height = fd.File(save_path+"results2/height.pvd")
    outfile_varphi = fd.File(save_path+"results2/err.pvd")
    outfile_sicks = fd.File(save_path+"results2/sicks.pvd")
    outfile_sickss = fd.File(save_path+"results2/sickss.pvd")
    fileE = save_path+"results2/potflow3dperenergy.txt"
    filEtaMax = save_path+"results2/potflow3dperetamax.txt"


#
outputE = open(fileE,"w")
# outputEtamax = open(filEtaMax,"w")

t = 0.0
i = 0.0

outfile_height.write(h_old, time=t)
outfile_psi.write(psi_f, time=t)
outfile_varphi.write(varphi, time=t)
    
# Not yet calculated: outfile_varphi.write(varphi, time=t)



tic = tijd.time()
nE0 = 0
while t <= 1.0*(t_end +dt): #  t_end + dt
    tt = format(t, '.3f')

    if nvpcase=="MMP":
        phi_combonl.solve()
        psimp, hmp, varphimp = result_mixedmp.split()
        psi_f.assign(2.0*psimp-psi_f)
        # err.interpolate(varphimp+phihat*psimp)
        varphi.interpolate(varphimp+phihat*psi_f)
         # update n+1 -> n
        h_old.assign(2.0*hmp-h_old) # update n+1 -> n
        EKin=fd.assemble( 0.5*( LwdWw2*(fac*H0+hmp)*(u0z + (psimp*phihat+varphimp).dx(0)-(1.0/(fac*H0+hmp))*(x[2]*hmp.dx(0))*(u0zpz +(psimp*phihat+varphimp).dx(2)))**2 \
                                  + (fac*H0+hmp)*(u0zpy + (psimp*phihat+varphimp).dx(1)-(1.0/(fac*H0+hmp))*(x[2]*hmp.dx(1))*(u0zpz + (psimp*phihat+varphimp).dx(2)))**2 \
                                  + (H0**2/(fac*H0+hmp))*(u0zpz +(psimp*phihat+varphimp).dx(2))**2 )* fd.dx(degree=vpoly)  )
        EPot=fd.assemble( gg*H0*(fac+hmp/H0)*(x[2]*(fac*H0+hmp)/H0-H0)* fd.dx(degree=vpoly) )
    
    elif nvpcase=="SV":
        phi_combonlsv.solve()
        psisv, varphisv = result_mixedsv.split()
        h_exprnl.solve()
        phin_exprnl.solve()
        psi_f.assign(psii)
        # Done later since needed in energy EKin: h_old.assign(h_new)
        # varphi.interpolate(varphisv+psi_f) # total velocity potential for plotting
        varphi.interpolate(varphisv+psisv*phihat) # total velocity potential for plotting to test take out phihat or not; in the end it should stay???
            

        # Energy monitoring: bit too frequent reduce
        
        EKin = fd.assemble( 0.25*( LwdWw2*(fac*H0+h_new)*(u0z + (psisv*phihat+varphisv).dx(0)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(0)+x[2]*h_new.dx(0))*((psisv*phihat+varphisv).dx(2)))**2 \
                                   + (fac*H0+h_new)*(u0zpy + (psisv*phihat+varphisv).dx(1)-(1.0/(fac*H0+h_new))*(H0*btopo.dx(1)+x[2]*h_new.dx(1))*((psisv*phihat+varphisv).dx(2)))**2 \
                                   + (H0**2/(fac*H0+h_new)) * (u0zpz + (psisv*phihat+varphisv).dx(2))**2 \
                                   + LwdWw2*(fac*H0+h_old)*(u0z + (psisv*phihat+varphisv).dx(0)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(0)+x[2]*h_old.dx(0))*((psisv*phihat+varphisv).dx(2)))**2 \
                                   + (fac*H0+h_old)*(u0zpy + (psisv*phihat+varphisv).dx(1)-(1.0/(fac*H0+h_old))*(H0*btopo.dx(1)+x[2]*h_old.dx(1))*((psisv*phihat+varphisv).dx(2)))**2 \
                                   + (H0**2/(fac*H0+h_old)) * (u0zpz + (psisv*phihat+varphisv).dx(2))**2  ) * fd.dx  )
        EPot = fd.assemble( (0.5*gg*H0*(fac+h_new/H0)*(x[2]*(fac*H0+h_new)/H0-H0) \
                            + 0.5*gg*H0*(fac+h_old/H0)*(x[2]*(fac*H0+h_old)/H0-H0) ) * fd.dx )
    if nE0==0:
        E0 = EKin+EPot
        PETSc.Sys.Print('energy =', E0)
    with h_old.dat.vec_ro as v:
         L_inf = v.max()[1]
    Etot = EKin+EPot-E0
    PETSc.Sys.Print('t =', t, EPot, EKin, Etot,L_inf)
        
    print("%.19f %.19f %.19f %.19f %.19f" %(t, EPot, EKin, Etot,L_inf),file=outputE)
    nE0+=int(1)
    t+= dt
    if nE0%50==0:
        outfile_height.write(h_old, time=t)
        outfile_psi.write(psi_f, time=t)
        outfile_varphi.write(varphi, time=t)

    
toc = tijd.time() - tic
#
if nprintout==1:
    print('Elapsed time (min):', toc/60)
else:
    PETSc.Sys.Print('Elapsed time (min):', toc/60)
    
outputE.close()
# outputEtamax.close()

with CheckpointFile(save_path+"data2d.h5", 'w') as afile2:
    afile2.save_mesh(mesh2d)  # optional
    afile2.save_function(h_old)
    afile2.save_function(psi_f)

# with CheckpointFile(save_path+"data.h5", 'w') as afile:
#     afile.save_mesh(mesh)  # optional
#     afile.save_function(varphi)
#     afile.save_function(EKin0)
#     afile.save_function(EPot0)



if nprintout==1:
    print('*************** PROGRAM ENDS ******************')
else:
    PETSc.Sys.Print('*************** PROGRAM ENDS ******************')
