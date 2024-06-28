####################################################################################
# Plain ball, SE, MMP and BE algorithms
####################################################################################
#
# GENERIC MODULES REQUIRED:
#
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve
from time import time
import os
import errno
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import time as tijd
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.mlab as mlab
import matplotlib.gridspec as gridspec
from math import pi, e
import importlib.util
import sys
import math
import pandas as pd
from pandas import read_excel
from pandas import to_datetime
from array import *
import calendar
import sys
import bisect
from scipy.optimize import least_squares
#  from numpy import exp, sin, cosh, tanh, log, random
from lmfit import minimize, fit_report, Parameters
# 
plt.close("all")

# Parameters
bb = 1.0
gamm = 100
bb = 2*0.34*np.sqrt(gamm)
tsize = 18 # font size of image title
tsize2 = 12
size = 16  # font size of image axes

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
Li1 = Li/3
nq = 1
Vt = 2.05
Isat = 0.02
Rl = nq*Vt/Isat
Rl1 = Rl/3
fac3  = 1.0
Rl1 = fac3*Rl
sigmm = 5.96*10**7
Rc = 8*a_rad*N/(sigmm*D_rad**2)
Ri = Rc
Ri1 = Ri/3
Rc1 = Rc/3
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
# Z00 = 0.205
GZ0 = Gapprox(Z00,-L/2,L/2,a_rad,Hm,alp,L,Zbar)
print('Zbar Z00, GZ0, gamma GZ0',Zbar,Z00,GZ0,gamma*GZ0)
plt.figure(1)
Nz1 = 200
ZZmin = 0.0
ZZmax = 0.2
ZZ1 = np.linspace(ZZmin,ZZmax,Nz1)
plt.plot(ZZ1,gamma*Gapprox(ZZ1,-L/2,L/2,a_rad,Hm,alp,L,Zbar),'-')
plt.plot(Z00,gamma*Gapprox(Z00,-L/2,L/2,a_rad,Hm,alp,L,Zbar),'xr')


# Parameters time stepping and such
Tperiod = 1.0
nTfac = 2*8
nforcefac = 2*6
nforcefac2 = 2*7
t_end = nTfac*Tperiod # time of simulation [s]
Tstartmeas = 0.0
dtt = np.minimum(0.01,0.005) # i.e. 
Nt = 1 # 
CFL = 0.5
dt = CFL*dtt # CFL*dtt
print('dtt=',dtt, t_end/dtt)
nplot = 400
dtmeet = t_end/nplot # 
tmeet = dtmeet
tmeas = np.linspace(0.0, t_end, nplot+1)
nvpcase = 1

##_________________  FIGURE SETTINGS __________________________##
print('Figure settings')

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)
# 
ax1.set_title(r'Forced buoy-wire model',fontsize=tsize2)   
ax1.set_ylabel(r'$Z(t)$ ',fontsize=size)
#  ax1.set_xlabel(r'$t$ ',fontsize=size)
ax2.set_ylabel(r'$W(t)$ ',fontsize=size)
# ax2.set_xlabel(r'$t$ ',fontsize=size)
ax3.set_ylabel(r'$Q(t)$ ',fontsize=size)
# ax3.set_xlabel(r'$t$ ',fontsize=size) ax4.set_ylabel(r'$P_Q(t), I(t)$ ',fontsize=size)
ax4.set_ylabel(r'$I(t)$ ',fontsize=size)
ax5.set_ylabel(r'$P_g(t)$ ',fontsize=size)
ax5.set_xlabel(r'$t$ ',fontsize=size)


tic = tijd.time()
Amp = 0.0653
sigma = 2*np.pi/Tperiod
facA = 0.4


# Initial conditons
t = 0
Z0 = 0.0
W0 = 0.0
PQ0 = 0.0
Q0 = 0.0
I0 = 0
Pg0 = Rl*I0**2
Pgmean = 0.0
Pgave0 = 0.0
Pgave = 0.0
nop = 1.0 # switches off the resistance damping

print('Time Loop starts.',np.sqrt(2*Z0))
nche = 0.5

while t <= 1.0*(t_end + dt): # W1,PQ1 1st then update Q1, Z1
    t = t+dt
    if t< nforcefac2*Tperiod:
        Force = Amp*np.cos(sigma*t)
        fhang = facA*Amp*np.cos(sigma*t)
        fhangdot = -sigma*facA*Amp*np.sin(sigma*t)
    else:
        Force = 0
        fhang = 0.0
        fhangdot = 0.0
        
    W1 = W0 - dt*gamma*GZ0*(PQ0+gamma*GZ0*Z0)/Li/Mm + dt*Force/Mm
    W1 = W0 - dt*gamma*GZ0*I0/Mm + dt*kspring*(fhang- Z0)/Mm
    
    # PQ1 = (PQ0 - dt*(nop*(Rc+Ri+Rl)/Li)*gamma*GZ0*Z0)/(1+dt*nop*(Rc+Ri+Rl)/Li) # does not work/get exact solution
    # PQ1 = (PQ0 - dt*(nop*(Rc+Ri+Rl)/Li)*(nche*PQ0+gamma*GZ0*Z0))/(1+dt*(1-nche)*nop*(Rc+Ri+Rl)/Li) # does not work/get exact solution
    # I1 =  I0*np.exp(-(Rc+Ri+Rl)/Li*dt) + gamma*GZ0*W1/Li*(Li/(Rc+Ri+Rl))*(1 - np.exp(-(Rc+Ri+Rl)/Li*dt)) # does not work/get exact solution
    # PQ1 = PQ0*np.exp(-(Rc+Ri+Rl)/Li*dt) - (nop*(Rc+Ri+Rl)/Li)*(gamma*GZ0*Z0)*(Li/(Rc+Ri+Rl))*(1-np.exp(-(Rc+Ri+Rl)/Li*dt)) # does not get exact sol
    # PQ1 = PQ0*np.exp(-(Rc+Ri+Rl)/Li*dt) - (nop*(Rc+Ri+Rl)/Li)*(gamma*GZ0*Z0)*dt # fails/blows up
    #  I1 = ( I0 + dt*gamma*GZ0*W1/Li -thfac*dt*nop*(Rc+Ri+Rl)*I0/Li )/(1 + (1-thfac)*dt*nop*(Rc+Ri+Rl)/Li)
    Z1 = Z0 + dt*W1
    Z1 = Z0 + dt*W1 + dt*fhangdot
    PQ1 = ( PQ0-dt*nche*((Rc+Ri+Rl)/Li)*(PQ0+gamma*GZ0*Z0)-dt*(nop*(Rc+Ri+Rl)/Li)*((1-nche)*gamma*GZ0*Z1) )/(1 + dt*(1-nche)*nop*(Rc+Ri+Rl)/Li) #
    # Q1 = Q0 + dt*(PQ1+gamma*GZ0*Z0)/Li # proper one gives weird large Q
    #
    Q1 = Q0 + dt*(PQ0+gamma*GZ0*Z0)/Li  # matching one with I 1-2-1
    I1 = (PQ1+gamma*GZ0*Z1)/Li
    
    ax1.plot([t-dt,t],[Z00+Z0,Z00+Z1],'-k')
    ax2.plot([t-dt,t],[W0,W1],'-k')
    ax3.plot([t-dt,t],[Q0,Q1],'-k')
    #  ax4.plot([t-dt,t],[PQ0,PQ1],'-k')
    ax4.plot([t-dt,t],[I0,I1],'-k')
    Pg1 = Rl*I1**2
    # ax5.plot([t-dt,t],[Pg0,Pg1],'-k')
    Pgmean = Pgmean + dt*Pg1
    Pgave = Pgmean/t
    ax5.plot([t-dt,t],[Pgave0,Pgave],'-k')
    W0 = W1
    Z0 = Z1
    PQ0 = PQ1
    Q0 = Q1
    I0 = I1
    Pg0 = Pg1
    Pgave0 = Pgave
# End while of time loop
PgavePQ2019 = Pgave

# Initial conditons
t = 0
Z0 = 0.0
W0 = 0.0
PQ0 = 0.0
Q0 = 0.0
I0 = 0
Pg0 = Rl*I0**2
Pgmean = 0.0
Pgave0 = 0.0
thfac = 0.5
while t <= 1.0*(t_end + dt): # W1, Q1 1st then update Z1, I1
    t = t+dt
    if t< nforcefac*Tperiod:
        Force = Amp*np.cos(sigma*t)
        fhang = facA*Amp*np.cos(sigma*t)
        fhangdot = -sigma*facA*Amp*np.sin(sigma*t)
    else:
        Force = 0.0
        fhang = 0.0
        fhangdot = 0.0
        
    W1 = W0 - dt*gamma*GZ0*I0/Mm + dt*Force/Mm
    W1 = W0 - dt*gamma*GZ0*I0/Mm + + dt*kspring*(fhang - Z0)/Mm
    Q1 = Q0 + dt*I0
    Z1 = Z0 + dt*W1
    Z1 = Z0 + dt*W1 + dt*fhangdot
    #
    I1 = ( I0 + dt*gamma*GZ0*(W1+fhangdot)/Li -thfac*dt*nop*(Rc+Ri+Rl)*I0/Li )/(1 + (1-thfac)*dt*nop*(Rc+Ri+Rl)/Li)
    # I1 = I0*np.exp(-(Rc+Ri+Rl)/Li*dt) + gamma*GZ0*W1/Li*(Li/(Rc+Ri+Rl))*(1 - np.exp(-(Rc+Ri+Rl)/Li*dt))
    PQ1 = Li*I1-gamma*GZ0*Z1
    
    ax1.plot([t-dt,t],[Z00+Z0,Z00+Z1],':b')
    ax2.plot([t-dt,t],[W0,W1],':b')
    ax3.plot([t-dt,t],[Q0,Q1],':b')
    #  ax4.plot([t-dt,t],[PQ0,PQ1],':b')
    ax4.plot([t-dt,t],[I0,I1],':b')
    Pg1 = Rl*I1**2
    # ax5.plot([t-dt,t],[Pg0,Pg1],':b')
    Pgmean = Pgmean + dt*Pg1
    Pgave = Pgmean/t
    ax5.plot([t-dt,t],[Pgave0,Pgave],':b')
    W0 = W1
    Z0 = Z1
    PQ0 = PQ1
    Q0 = Q1
    I0 = I1
    Pg0 = Pg1
    Pgave0 = Pgave
# End while of time loop
PgaveI2021 = Pgave

Pgexact = 0.5*Rl*(gamma*GZ0)**2*Amp**2/(((gamma*GZ0)**2-Li*Mm*sigma**2)**2 + (Mm*(Rc+Ri+Rl)*sigma)**2)
print('PG: exact, 2019PQ, 2021ICL',Pgexact,PgavePQ2019,PgaveI2021)
# old or check:
ax5.plot([0,t_end],[Pgexact,Pgexact],'-r')

# Initial conditons nonlinear case with iterations for I^n+1
t = 0
Z0 = Z00 
W0 = 0.0
PQ0 = 0.0
Q0 = 0.0
I0 = 0
Q02 = 0.0
I02 = 0
Q03 = 0.0
I03 = 0
Pg0 = Rl*I0**2
Pgmean = 0.0
Pgave0 = 0.0
thfac = 0.5
iiN = 12
smallfac = 10**(-14) # old
smallfac = 10**(-14)
smallfac2 = 10**(-10) # old smallfac2 = 10**(-8)  #
while t <= 1.0*(t_end + dt): # W1, Q1 1st then update Z1, I1
    t = t+dt
    if t< nforcefac*Tperiod:
        Force = Mm*grav + Amp*np.cos(sigma*t)
        fhang = facA*Amp*np.cos(sigma*t)
        fhangdot = -sigma*facA*Amp*np.sin(sigma*t)
    else:
        Force = Mm*grav
        fhang = 0.0
        fhangdot = 0.0

    GZ0 = Gapprox(Z0,-L/2,-L/6,a_rad,Hm,alp,L,Zbar)
    GZ02 = Gapprox(Z0,-L/6,L/6,a_rad,Hm,alp,L,Zbar)
    GZ03 = Gapprox(Z0,L/6,L/2,a_rad,Hm,alp,L,Zbar)
    # W1 = W0 - dt*grav - dt*gamma*GZ0*I0/Mm + dt*Force/Mm # + dt*kspring*(Force  + Z0 - Zbar)/Mm
    W1 = W0 - dt*gamma*GZ0*I0/Mm - dt*gamma*GZ02*I02/Mm - dt*gamma*GZ03*I03/Mm + dt*kspring*(fhang  + Zbar - Z0)/Mm
    Q1 = Q0 + dt*I0
    Q2 = Q02 + dt*I02
    Q3 = Q03 + dt*I03
    Z1 = Z0 + dt*(W1+fhangdot)
    #
    GZ1 = Gapprox(Z1,-L/2,-L/6,a_rad,Hm,alp,L,Zbar)
    GZ2 = Gapprox(Z1,-L/6,L/6,a_rad,Hm,alp,L,Zbar)
    GZ3 = Gapprox(Z1,L/6,L/2,a_rad,Hm,alp,L,Zbar)
    # I1 = ( I0 + dt*gamma*GZ1*W1/Li -thfac*dt*nop*(Rc+Ri)*I0/Li )/(1 + (1-thfac)*dt*nop*(Rc+Ri+Rl)/Li)
    # I1 = ( I0 + dt*gamma*GZ1*W1/Li -thfac*dt*nop*(Rc+Ri+Rl)*I0/Li )/(1 + (1-thfac)*dt*nop*(Rc+Ri+Rl)/Li)
    # 1st-order Strang splitting step 1 Ii-equations without Thackley load
    # 1st-order Strang splitting step 2 I1-equations with only Thackley load part; solve for P1, P2, P3
    if iiN>0:
        diffe = 1
        diffe2 = 1
        diffe3 = 1
        I0g = I0 # 1st guess used as iterate for I1
        I02g = I02 # 1st guess used as iterate for I2
        I03g = I03 # 1st guess used as iterate for I3
        cc = dt*nq*Vt/Li1
        # for ii in range(0, iiN):
        while (diffe+diffe2+diffe3>smallfac2):
            bb = dt*nop*(Rc1+Ri1)/Li1
            dd = (np.abs(I0g)+np.abs(I0)+np.abs(I02g)+np.abs(I02)+np.abs(I03g)+np.abs(I03))
            dd1 = (np.abs(I0g)+np.abs(I0))
            dd2 = (np.abs(I02g)+np.abs(I02))
            dd3 = (np.abs(I03g)+np.abs(I03))
            # New 10-06-2024 above one works next one does not converge; positivity of both required.
            # dd = (np.abs(I0g+I0)+np.abs(I02g+I02)+np.abs(I03g+I03))
            # dd1 = (np.abs(I0g+I0))
            # dd2 = (np.abs(I02g+I02))
            # dd3 = (np.abs(I03g+I03))
            if dd1<smallfac:
                ee1 = 0.0
            else:
                ee1 = cc*np.log(1+0.5*(dd)/Isat)/dd1
            if dd2<smallfac:
                ee2 = 0.0
            else:
                ee2 = cc*np.log(1+0.5*(dd)/Isat)/dd2
            if dd3<smallfac:
                ee3 = 0.0
            else:
                ee3 = cc*np.log(1+0.5*(dd)/Isat)/dd3
                
            I1 = ((1-ee1-thfac*bb)*I0+dt*gamma*GZ1*(W1+fhangdot)/Li1 )/(1+ee1+(1-thfac)*bb)
            I2 = ((1-ee2-thfac*bb)*I02+dt*gamma*GZ2*(W1+fhangdot)/Li1 )/(1+ee2+(1-thfac)*bb)
            I3 = ((1-ee3-thfac*bb)*I03+dt*gamma*GZ3*(W1+fhangdot)/Li1 )/(1+ee3+(1-thfac)*bb)
            diffe  = np.abs(I1-I0g)
            diffe2 = np.abs(I2-I02g)
            diffe3 = np.abs(I3-I03g)
            I0g = I1
            I02g = I2
            I03g = I3
            
        print('I1-I0g, I2-I02g, I3-I03g', diffe, diffe2, diffe3)
    else: # "linear" damping 4 Thackley load
        cc = dt*nq*Vt/Li1
        dd = (2*Isat)
        I1 = ( I0  + dt*gamma*GZ1*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1+Rl1)*I0/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1+Rl1)/Li1)
        I2 = ( I02 + dt*gamma*GZ2*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1+Rl1)*I02/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1+Rl1)/Li1)
        I3 = ( I03 + dt*gamma*GZ3*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1+Rl1)*I03/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1+Rl1)/Li1)
        nchallo = 1
        if nchallo==0:
            I1 = ( I0 + dt*gamma*GZ1*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1)*I0/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1)/Li1)
            I0 = 2*np.abs(I1) # 1st guess in 2nd step then used as iterate for P1
            I1mabs = np.abs(I1) # end of 1st split step absolute value
            I2 = ( I02 + dt*gamma*GZ2*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1)*I02/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1)/Li1)
            I02 = 2*np.abs(I2)
            I2mabs = np.abs(I2)
            I3 = ( I03 + dt*gamma*GZ3*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1)*I03/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1)/Li1)
            I03 = 2*np.abs(I3)
            I3mabs = np.abs(I3)
            I1 = ( I0  + dt*gamma*GZ1*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1)*I0/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1)/Li1)
            I2 = ( I02 + dt*gamma*GZ2*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1)*I02/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1)/Li1)
            I3  = ( I03 + dt*gamma*GZ3*(W1+fhangdot)/Li1 -thfac*dt*nop*(Rc1+Ri1)*I03/Li1 )/(1 + (1-thfac)*dt*nop*(Rc1+Ri1)/Li1)
            ee = cc/dd
            F1 = I1mabs-ee*(I1mabs+I2mabs+I3mabs)
            F2 = I2mabs-ee*(I1mabs+I2mabs+I3mabs)
            F3 = I3mabs-ee*(I1mabs+I2mabs+I3mabs)
            P1 = ( (1+2*ee)*F1-ee*F2-ee*F3)/(1+3*ee)
            P2 = (-ee*F1+(1+2*ee)*F2-ee*F3)/(1+3*ee)
            P3 = (-ee*F1-ee*F2+(1+2*ee)*F3)/(1+3*ee)
            if P1<0 or P2<0 or P3<0:
                print(' P1, P2 or P3 <0')
                print('hallo ', P1/I1, P2/I2, P3/I3)
                I1 = np.sign(I1)*P1
                I2 = np.sign(I2)*P2
                I3 = np.sign(I3)*P3
  
    # I1 = ( I0 + dt*gamma*GZ1*W1/Li -thfac*dt*nop*(Rc+Ri+Rl)*I0/Li )/(1 + (1-thfac)*dt*nop*(Rc+Ri+Rl)/Li)
    # I1 = I0*np.exp(-(Rc+Ri+Rl)/Li*dt) + gamma*GZ0*W1/Li*(Li/(Rc+Ri+Rl))*(1 - np.exp(-(Rc+Ri+Rl)/Li*dt))
    PQ1 = Li*I1-gamma*GZ0*Z1
    # J1 = Inp1+I0
    
    ax1.plot([t-dt,t],[Z0,Z1],'--r')
    ax2.plot([t-dt,t],[W0,W1],'--r')
    ax3.plot([t-dt,t],[Q0,Q1],'--r')
    #  ax4.plot([t-dt,t],[PQ0,PQ1],'--r')
    ax4.plot([t-dt,t],[I0,I1],'--r')
    ax4.plot([t-dt,t],[I02,I2],'--r')
    ax4.plot([t-dt,t],[I03,I3],'--r')
    Pg1 = (np.abs(I1)+np.abs(I2)+np.abs(I3))*nq*Vt*np.log(1+(np.abs(I1)+np.abs(I2)+np.abs(I3))/Isat)
    # ax5.plot([t-dt,t],[Pg0,Pg1],'--r')
    Pgmean = Pgmean + dt*Pg1
    Pgave = Pgmean/t
    ax5.plot([t-dt,t],[Pgave0,Pgave],'--r')
    W0 = W1
    Z0 = Z1
    PQ0 = PQ1
    Q0 = Q1
    Q02 = Q2
    Q03 = Q3
    I0 = I1
    I02 = I2
    I03 = I3
    Pg0 = Pg1
    Pgave0 = Pgave
# End while of time loop
PgaveI2021 = Pgave

Pgexact = 0.5*Rl*(gamma*GZ0)**2*Amp**2/(((gamma*GZ0)**2-Li*Mm*sigma**2)**2 + (Mm*(Rc+Ri+Rl)*sigma)**2)
print('PG: exact, 2019PQ, 2021ICL',Pgexact,PgavePQ2019,PgaveI2021)
# old/check
ax5.plot([0,t_end],[Pgexact,Pgexact],'-.g')


toc = tijd.time() - tic
print('Elapsed time (min):', toc/60)

plt.show() 
print('*************** PROGRAM ENDS ******************')
