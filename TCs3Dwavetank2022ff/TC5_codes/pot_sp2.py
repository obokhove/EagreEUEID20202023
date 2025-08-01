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

# op2.init()
# parameters["coffee"]["O2"] = False

# parameters in SI units REMIS error of polynomials
gg = 9.81  # gravitational acceleration [m/s^2]

# water domain and discretisation parameters
# nic = "linearw" # choice initial condition
nic = "SP2" # SP2: two-soliton case 
nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases;
nphihatz = "Unity" # "Unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.

nCG = 2       # order of CG. Choices: ["2","4"] 
multiple=1    # a resolution factor in spatial domain.  1 is basic. 2 is dense. 
multiple2=1   # a resolution factor in temporal domain. 1 is basic. 2 is dense.
nprintout = 0 # print out on screen or in file; 1: on screen: 0: PETSC version
     
if nic=="SP2": # two-soliton case made x-periodic
    eps = 0.05 # WW2022-paper setting
    muu = eps**2
    muu_pfe=muu
    Lz= 20
    H0 = Lz
    y2hat = 20*0.65*(muu)**.5*(2**.5/3)**(2/3)/eps
    y1hat = -20*0.35*(muu)**.5*(2**.5/3)**(2/3)/eps
    Y2KPE = (eps/np.sqrt(muu))*(3/np.sqrt(2))**(2/3)*y2hat
    Ly = (H0/np.sqrt(muu_pfe))*(y2hat-y1hat)
    yy2 = (H0/np.sqrt(muu_pfe))*y2hat
    yy1 = (H0/np.sqrt(muu_pfe))*y1hat
    
    delt = 10**(-5)
    k1 = -(5/((4/3)**(1/3))/6)**.5
    # k1 = -1
    k2 = k1+delt
    k3 = delt
    k4 = -k1
    
    a0 = 1
    b0 = 1
    # Y2=3.5
    Y2=3.0
    t1=0
    tau=t1*eps*(2*eps/muu)**(1/2)
    # yhatstar = -(np.sqrt(muu)/eps)*(np.sqrt(2)/3)**(2/3)*np.log(2)/k4**2
# 
    # xx2hat = k4*np.sqrt(eps)*(3/np.sqrt(2))**(1/3)*y2hat  + np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*np.log(2)/k4
    
    X1KPE = -np.log((k2-k1)/(k3-k1))/(k2-k3)-(k2+k3)*Y2+(k2**2+k2*k3+k3**2)*tau
    X2KPE = -np.log((k4-k2)/(k4-k3))/(k2-k3)-(k2+k3)*Y2+(k2**2+k2*k3+k3**2)*tau
    x2hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X2KPE)
    x1hat = (np.sqrt(muu/eps)*(np.sqrt(2)/3)**(1/3)*X1KPE)
    yfac = 1.0
    xx1 = yfac*(H0/np.sqrt(muu_pfe))*x1hat
    xx2 = yfac*(H0/np.sqrt(muu_pfe))*x2hat
    Lx = xx2-xx1
    # value = ((2/9)**(1/6)*0.25/(0.87013))**2
    
    PETSc.Sys.Print('Lx,Ly,x1hat,x2hat,y1hat,y2hat: ',Lx,Ly,x1hat,x2hat,y1hat,y2hat)
    # PETSc.Sys.Print('yhatstk4: ',yhatstar,k4,value,np.log(2))
    
    nx = int(np.round(multiple*20*(x2hat-x1hat)*4/nCG))
    ny = int(np.round(multiple*20*(y2hat-y1hat)*4/nCG))
    nz = 4
    # function space order horizontal
    nCGvert = 2 # function space order vertical
    nvpcase = "MMP" # MMP=modified midpoint VP time discretisation in case of making more cases; SV=Stormer-Verlet
    nphihatz = "Unity" # "unity": phihat=1.0; 1: "GLL1" 1st GLL at nCGvert, etc.


#________________________ MESH  _______________________#


mesh2d = PeriodicRectangleMesh(nx, ny, Lx, Ly, direction="x",
                               quadrilateral=True, reorder=None,distribution_parameters=None,diagonal=None,comm=COMM_WORLD,name='mesh2d') # xx1, xx2, yy1, yy2
mesh = fd.ExtrudedMesh(mesh2d, nz, layer_height=Lz/nz, extrusion_type='uniform',name='mesh')
# x, y, z = fd.SpatialCoordinate(mesh)
x = mesh.coordinates
top_id = 'top'
# 



t = 0

fac = 1.0 # Used to split h=H0+eta in such in way that we can switch to solving h (fac=0.0) or eta (fac=1.0)
# u0 = 0.0 # periodicity factor; default 0

if nic=="SP2": # SP2 solitons periodic dimensional; extend VP? Define eta and phi
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
   
    theta11 =  k1*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx11-xs-np.sqrt(gg*H0)*t0) + k1**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k1**3*tau
    theta12 =  k2*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx11-xs-np.sqrt(gg*H0)*t0) + k2**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k2**3*tau
    theta13 =  k3*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx11-xs-np.sqrt(gg*H0)*t0) + k3**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k3**3*tau
    theta14 =  k4*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx11-xs-np.sqrt(gg*H0)*t0) + k4**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k4**3*tau
   
    KK1 = (k2-k1)*exp(theta11+theta12)+\
        a0*(k3-k1)*exp(theta11+theta13)+\
            b0*(k4-k2)*exp(theta12+theta14)+\
        a0*b0*(k4-k3)*exp(theta13+theta14)
    KKX1 = (k1+k2)*(k2-k1)*exp(theta11+theta12)+\
        (k1+k3)*a0*(k3-k1)*exp(theta11+theta13)+\
        (k2+k4)*b0*(k4-k2)*exp(theta12+theta14)+\
        (k3+k4)*a0*b0*(k4-k3)*exp(theta13+theta14)
    KKXX1 = (k1+k2)**2*(k2-k1)*exp(theta11+theta12)+\
        (k1+k3)**2*a0*(k3-k1)*exp(theta11+theta13)+\
        (k2+k4)**2*b0*(k4-k2)*exp(theta12+theta14)+\
        (k3+k4)**2*a0*b0*(k4-k3)*exp(theta13+theta14)
    KKXXX1 = (k1+k2)**3*(k2-k1)*exp(theta11+theta12)+\
        (k1+k3)**3*a0*(k3-k1)*exp(theta11+theta13)+\
        (k2+k4)**3*b0*(k4-k2)*exp(theta12+theta14)+\
        (k3+k4)**3*a0*b0*(k4-k3)*exp(theta13+theta14)
    KKY1 = (k1**2+k2**2)*(k2-k1)*exp(theta11+theta12)+\
        (k1**2+k3**2)*a0*(k3-k1)*exp(theta11+theta13)+\
        (k2**2+k4**2)*b0*(k4-k2)*exp(theta12+theta14)+\
        (k3**2+k4**2)*a0*b0*(k4-k3)*exp(theta13+theta14)
    KKXYY1 = (k1**2+k2**2)**2*(k1+k2)*(k2-k1)*exp(theta11+theta12)+\
        (k1**2+k3**2)**2*a0*(k1+k3)*(k3-k1)*exp(theta11+theta13)+\
        (k2**2+k4**2)**2*(k2+k4)*b0*(k4-k2)*exp(theta12+theta14)+\
        (k3**2+k4**2)**2*(k3+k4)*a0*b0*(k4-k3)*exp(theta13+theta14)
    KKYY1 = (k1**2+k2**2)**2*(k2-k1)*exp(theta11+theta12)+\
        (k1**2+k3**2)**2*a0*(k3-k1)*exp(theta11+theta13)+\
        (k2**2+k4**2)**2*b0*(k4-k2)*exp(theta12+theta14)+\
        (k3**2+k4**2)**2*a0*b0*(k4-k3)*exp(theta13+theta14)
    KKXY1 = (k1**2+k2**2)*(k1+k2)*(k2-k1)*exp(theta11+theta12)+\
        (k1**2+k3**2)*a0*(k1+k3)*(k3-k1)*exp(theta11+theta13)+\
        (k2**2+k4**2)*(k2+k4)*b0*(k4-k2)*exp(theta12+theta14)+\
        (k3**2+k4**2)*(k3+k4)*a0*b0*(k4-k3)*exp(theta13+theta14)
    GX1 = 2*Fx**2*( KKXXX1/KK1-3*KKXX1*KKX1/KK1**2+2*KKX1**3/KK1**3 )
    GY1 = 2*Fy**2*( KKXYY1/KK1-2*KKXY1*KKY1/KK1**2-KKX1*KKYY1/KK1**2+2*KKX1*(KKY1)**2/KK1**3 )
    etax1y = 2.0*eps*H0*((4/3)**(1/3))*( KKXX1/KK1 -(KKX1/KK1)**2 )
    psix1y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu_pfe))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1-0.5*muu_pfe*((H0+0.0*etax1y)/H0)**2*(GX1+GY1))
    psix1yz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu_pfe))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1-0.5*muu_pfe*((x[2])/H0)**2*(GX1+GY1))
    # psix1y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX1/KK1)

    # xx2 used x[1]=y function ; can this be automated via general case?
    theta21 =  k1*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx22-xs-np.sqrt(gg*H0)*t0) + k1**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k1**3*tau
    theta22 =  k2*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx22-xs-np.sqrt(gg*H0)*t0) + k2**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k2**3*tau
    theta23 =  k3*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx22-xs-np.sqrt(gg*H0)*t0) + k3**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k3**3*tau
    theta24 =  k4*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(xx22-xs-np.sqrt(gg*H0)*t0) + k4**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k4**3*tau
   
    KK2 = (k2-k1)*exp(theta21+theta22)+\
        a0*(k3-k1)*exp(theta21+theta23)+\
            b0*(k4-k2)*exp(theta22+theta24)+\
        a0*b0*(k4-k3)*exp(theta23+theta24)
    KKX2 = (k1+k2)*(k2-k1)*exp(theta21+theta22)+\
        (k1+k3)*a0*(k3-k1)*exp(theta21+theta23)+\
        (k2+k4)*b0*(k4-k2)*exp(theta22+theta24)+\
        (k3+k4)*a0*b0*(k4-k3)*exp(theta23+theta24)
    KKXX2 = (k1+k2)**2*(k2-k1)*exp(theta21+theta22)+\
        (k1+k3)**2*a0*(k3-k1)*exp(theta21+theta23)+\
        (k2+k4)**2*b0*(k4-k2)*exp(theta22+theta24)+\
        (k3+k4)**2*a0*b0*(k4-k3)*exp(theta23+theta24)
    KKXXX2 = (k1+k2)**3*(k2-k1)*exp(theta21+theta22)+\
        (k1+k3)**3*a0*(k3-k1)*exp(theta21+theta23)+\
        (k2+k4)**3*b0*(k4-k2)*exp(theta22+theta24)+\
        (k3+k4)**3*a0*b0*(k4-k3)*exp(theta23+theta24)
    KKY2 = (k1**2+k2**2)*(k2-k1)*exp(theta21+theta22)+\
        (k1**2+k3**2)*a0*(k3-k1)*exp(theta21+theta23)+\
        (k2**2+k4**2)*b0*(k4-k2)*exp(theta22+theta24)+\
        (k3**2+k4**2)*a0*b0*(k4-k3)*exp(theta23+theta24)
    KKXYY2 = (k1**2+k2**2)**2*(k1+k2)*(k2-k1)*exp(theta21+theta22)+\
        (k1**2+k3**2)**2*a0*(k1+k3)*(k3-k1)*exp(theta21+theta23)+\
        (k2**2+k4**2)**2*(k2+k4)*b0*(k4-k2)*exp(theta22+theta24)+\
        (k3**2+k4**2)**2*(k3+k4)*a0*b0*(k4-k3)*exp(theta23+theta24)
    KKYY2 = (k1**2+k2**2)**2*(k2-k1)*exp(theta21+theta22)+\
        (k1**2+k3**2)**2*a0*(k3-k1)*exp(theta21+theta23)+\
        (k2**2+k4**2)**2*b0*(k4-k2)*exp(theta22+theta24)+\
        (k3**2+k4**2)**2*a0*b0*(k4-k3)*exp(theta23+theta24)
    KKXY2 = (k1**2+k2**2)*(k1+k2)*(k2-k1)*exp(theta21+theta22)+\
        (k1**2+k3**2)*a0*(k1+k3)*(k3-k1)*exp(theta21+theta23)+\
        (k2**2+k4**2)*(k2+k4)*b0*(k4-k2)*exp(theta22+theta24)+\
        (k3**2+k4**2)*(k3+k4)*a0*b0*(k4-k3)*exp(theta23+theta24)
    GX2 = 2*Fx**2*( KKXXX2/KK2-3*KKXX2*KKX2/KK2**2+2*KKX2**3/KK2**3 )
    GY2 = 2*Fy**2*( KKXYY2/KK2-2*KKXY2*KKY2/KK2**2-KKX2*KKYY2/KK2**2+2*KKX2*(KKY2)**2/KK2**3 )
    etax2y = eps*H0*((4/3)**(1/3))*2.0*( KKXX2/KK2 -(KKX2/KK2)**2 )
    psix2y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu_pfe))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2-0.5*muu_pfe*((H0+0.0*etax2y)/H0)**2*(GX2+GY2))
    psix2yz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu_pfe))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2-0.5*muu_pfe*((x[2])/H0)**2*(GX2+GY2))
    # psix2y = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX2/KK2)

    U0y = (psix2y-psix1y)/(xx22-xx11)
   
    U0yxc0y = U0y*x[0] # (U0y*x[0]+c0y)
    sicko = U0y*x[0]
    # u0 = U0yxc0y.dx(0)
    u0py = U0yxc0y.dx(1)*x[0]
    
    u0z = (psix2yz-psix1yz)/(xx22-xx11)
    
    # U0yxc0yz = u0z*x[0]+0.0*x[1]
    U0yxc0yz = u0z*x[0]+0.0*x[1]
    u0zpy = U0yxc0yz.dx(1)
    u0zpz = U0yxc0yz.dx(2)

    # X = ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(x[0]-np.sqrt(gg*H0)*t0) ; Y = ( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*x[1]
    theta1 =  k1*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(x[0]-xs-np.sqrt(gg*H0)*t0) + k1**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k1**3*tau
    theta2 =  k2*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(x[0]-xs-np.sqrt(gg*H0)*t0) + k2**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k2**3*tau
    theta3 =  k3*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(x[0]-xs-np.sqrt(gg*H0)*t0) + k3**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k3**3*tau
    theta4 =  k4*( ((muu_pfe**.5/H0)*(eps/muu)**.5)*(3/np.sqrt(2))**(1/3) )*(x[0]-xs-np.sqrt(gg*H0)*t0) + k4**2*( ((muu_pfe**.5/H0)*eps*(1/muu)**.5)*(3/np.sqrt(2))**(2/3) )*(x[1]-ys)\
        -k4**3*tau
   
    
    KK = (k2-k1)*exp(theta1+theta2)+\
        a0*(k3-k1)*exp(theta1+theta3)+\
            b0*(k4-k2)*exp(theta2+theta4)+\
        a0*b0*(k4-k3)*exp(theta3+theta4)
    KKX = (k1+k2)*(k2-k1)*exp(theta1+theta2)+\
        (k1+k3)*a0*(k3-k1)*exp(theta1+theta3)+\
        (k2+k4)*b0*(k4-k2)*exp(theta2+theta4)+\
        (k3+k4)*a0*b0*(k4-k3)*exp(theta3+theta4)
    KKXX = (k1+k2)**2*(k2-k1)*exp(theta1+theta2)+\
        (k1+k3)**2*a0*(k3-k1)*exp(theta1+theta3)+\
        (k2+k4)**2*b0*(k4-k2)*exp(theta2+theta4)+\
        (k3+k4)**2*a0*b0*(k4-k3)*exp(theta3+theta4)
    KKXXX = (k1+k2)**3*(k2-k1)*exp(theta1+theta2)+\
        (k1+k3)**3*a0*(k3-k1)*exp(theta1+theta3)+\
        (k2+k4)**3*b0*(k4-k2)*exp(theta2+theta4)+\
        (k3+k4)**3*a0*b0*(k4-k3)*exp(theta3+theta4)
    KKY = (k1**2+k2**2)*(k2-k1)*exp(theta1+theta2)+\
        (k1**2+k3**2)*a0*(k3-k1)*exp(theta1+theta3)+\
        (k2**2+k4**2)*b0*(k4-k2)*exp(theta2+theta4)+\
        (k3**2+k4**2)*a0*b0*(k4-k3)*exp(theta3+theta4)
    KKXYY = (k1**2+k2**2)**2*(k1+k2)*(k2-k1)*exp(theta1+theta2)+\
        (k1**2+k3**2)**2*(k3+k1)*a0*(k3-k1)*exp(theta1+theta3)+\
        (k2**2+k4**2)**2*(k4+k2)*b0*(k4-k2)*exp(theta2+theta4)+\
        (k3**2+k4**2)**2*(k4+k3)*a0*b0*(k4-k3)*exp(theta3+theta4)
    KKYY = (k1**2+k2**2)**2*(k2-k1)*exp(theta1+theta2)+\
        (k1**2+k3**2)**2*a0*(k3-k1)*exp(theta1+theta3)+\
        (k2**2+k4**2)**2*b0*(k4-k2)*exp(theta2+theta4)+\
        (k3**2+k4**2)**2*a0*b0*(k4-k3)*exp(theta3+theta4)
    KKXY = (k1**2+k2**2)*(k1+k2)*(k2-k1)*exp(theta1+theta2)+\
        (k1**2+k3**2)*a0*(k1+k3)*(k3-k1)*exp(theta1+theta3)+\
        (k2**2+k4**2)*(k2+k4)*b0*(k4-k2)*exp(theta2+theta4)+\
        (k3**2+k4**2)*(k3+k4)*a0*b0*(k4-k3)*exp(theta3+theta4)
    GX = 2*Fx**2*( KKXXX/KK-3*KKXX*KKX/KK**2+2*KKX**3/KK**3 )
    GY = 2*Fy**2*( KKXYY/KK-2*KKXY*KKY/KK**2-KKX*KKYY/KK**2+2*KKX*(KKY)**2/KK**3 )
    # KKXXX = k4**2*KKX # k4**4*(np.exp(theta4)-np.exp(theta1))
    # KKXYY = k4**2*KKXY # k4**6*(np.exp(theta4)-np.exp(theta1))
    eta_exact_expr = 2.0*eps*H0*((4/3)**(1/3))*( KKXX/KK -(KKX/KK)**2 )
    psi_exact_exprH0 = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu_pfe))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu_pfe*((H0+eta_exact_expr)/H0)**2*(GX+GY))-(U0y*x[0])
    psi_exact_exprz = ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu_pfe))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu_pfe*((x[2])/H0)**2*(GX+GY))-(u0z*x[0])
    sickofit =         ((eps*H0*np.sqrt(gg*H0))/np.sqrt(muu_pfe))*np.sqrt(eps)*(4*np.sqrt(2)/9)**(1/3)*(2*KKX/KK-0.5*muu_pfe*((H0+eta_exact_expr)/H0)**2*(GX+GY))-(U0y*x[0])
    
    btopoexpr = 0.0*psi_exact_exprH0 # no topography
    ttilde = 50 # time units used for SP2 in BLE
    # ttilde = 2
    t_end = ttilde*(H0/np.sqrt(gg*H0*muu_pfe)) # dimensionless BLE end-time of BokhoveKalogirou2016 times time-scaling factor
    # t2hat = eps*np.sqrt(2*eps/muu**2) # BLE to PFE time scaling
    # t_end = ttilde*t2hat
    Nt = ttilde*100*multiple2
    # Nt = 80*10
    dt = t_end/Nt # Somehow depends on soliton speed; so dx/U0 with U0=x/t dimensional from argument soliton; see dtt
    CFL = 0.5
    cc = 1
    dtt = CFL*(Lx/nx)*6.0/((1+eps*cc)*np.sqrt(gg*H0))    
    
    nplot = 10
    tijd.sleep(0.1)
    



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


# 
    

###### OUTPUT FILES and initial PLOTTING ##########

save_path =  "data/"

if not os.path.exists(save_path):
    os.makedirs(save_path) 


outfile_psi = fd.File(save_path+"psi.pvd")
outfile_height = fd.File(save_path+"height.pvd")
outfile_varphi = fd.File(save_path+"varphi.pvd")
fileE = save_path+"potflow3dperenergy.txt"



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

    
    phi_combonl.solve()
    psimp, hmp, varphimp = result_mixedmp.subfunctions
    psi_f.assign(2.0*psimp-psi_f)
    # err.interpolate(varphimp+phihat*psimp)
    varphi.interpolate(varphimp+phihat*psi_f)
      # update n+1 -> n
    h_old.assign(2.0*hmp-h_old) # update n+1 -> n
    EKin=fd.assemble( 0.5*( LwdWw2*(fac*H0+hmp)*(u0z + (psimp*phihat+varphimp).dx(0)-(1.0/(fac*H0+hmp))*(x[2]*hmp.dx(0))*(u0zpz +(psimp*phihat+varphimp).dx(2)))**2 \
                              + (fac*H0+hmp)*(u0zpy + (psimp*phihat+varphimp).dx(1)-(1.0/(fac*H0+hmp))*(x[2]*hmp.dx(1))*(u0zpz + (psimp*phihat+varphimp).dx(2)))**2 \
                              + (H0**2/(fac*H0+hmp))*(u0zpz +(psimp*phihat+varphimp).dx(2))**2 )* fd.dx(degree=vpoly)  )
    EPot=fd.assemble( gg*H0*(fac+hmp/H0)*(x[2]*(fac*H0+hmp)/H0-H0)* fd.dx(degree=vpoly) )
    
    
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
    if nE0%100==0:
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

# with CheckpointFile(save_path+"data2d.h5", 'w') as afile2:
#     afile2.save_mesh(mesh2d)  # optional
#     afile2.save_function(h_old)
#     afile2.save_function(psi_f)



if nprintout==1:
    print('*************** PROGRAM ENDS ******************')
else:
    PETSc.Sys.Print('*************** PROGRAM ENDS ******************')
