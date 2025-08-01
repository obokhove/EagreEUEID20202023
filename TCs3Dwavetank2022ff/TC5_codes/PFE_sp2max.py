import numpy as np
import pandas as pd
import os.path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt



fileE = 'C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/delt10-5nt200/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []

for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[4]))
    # AA_data.append(float(words[2]))


tt200ndt=np.array(tijd)
max200ndt=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

long=len(max200ndt)

qq=int(200)
lqq=int(long/qq)
max1200ndt=np.zeros((lqq+1))
max1200ndt[0]=max200ndt[0]


for jj in range(lqq):
    max1200ndt[jj+1]=sum(max200ndt[jj*qq:(jj+1)*qq])/qq
    
    # print(max_pfe[jj+1])
    # input('sfdg')
tt1200ndt=np.linspace(tt200ndt[0],tt200ndt[-1],lqq+1)




    

fileE = 'C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/delt10-5nt100/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []

for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[4]))
    # AA_data.append(float(words[2]))


tt=np.array(tijd)
maxx=np.array(max_data)

long=len(maxx)

qq=int(100)
lqq=int(long/qq)
max1=np.zeros((lqq+1))
max1[0]=maxx[0]

for jj in range(lqq):
    max1[jj+1]=sum(maxx[jj*qq:(jj+1)*qq])/qq
    # print(max_pfe[jj+1])
    # input('sfdg')
tt1=np.linspace(tt[0],tt[-1],lqq+1)

# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()


fileE = 'C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/delt10-5muti10/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []

for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[4]))
    # AA_data.append(float(words[2]))


tt10multi=np.array(tijd)
max10multi=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

long=len(max10multi)

qq=int(100)
lqq=int(long/qq)
max110multi=np.zeros((lqq+1))
max110multi[0]=max10multi[0]

for jj in range(lqq):
    max110multi[jj+1]=sum(max10multi[jj*qq:(jj+1)*qq])/qq
    # print(max_pfe[jj+1])
    # input('sfdg')
tt110multi=np.linspace(tt10multi[0],tt10multi[-1],lqq+1)


fileE = 'C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/delt10-5nt100cg4/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []

for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[4]))
    # AA_data.append(float(words[2]))


tt4cg=np.array(tijd)
max4cg=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

long=len(max4cg)

qq=int(100)
lqq=int(long/qq)
max14cg=np.zeros((lqq+1))
max14cg[0]=max4cg[0]

for jj in range(lqq):
    max14cg[jj+1]=sum(max4cg[jj*qq:(jj+1)*qq])/qq
    # print(max_pfe[jj+1])
    # input('sfdg')
tt14cg=np.linspace(tt4cg[0],tt4cg[-1],lqq+1)

df = pd.read_csv("C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/ble_del10-5cg4/max.csv")
maxble = df.to_numpy()

long=len(maxble)

qq=int(100)
lqq=int(long/qq)
max1ble=np.zeros((lqq+1))
max1ble[0]=maxble[0]

for jj in range(lqq):
    max1ble[jj+1]=sum(maxble[jj*qq:(jj+1)*qq])/qq


plt.figure(figsize=(10,6))
plt.plot(tt1,max1,label=r'$CG2/\Delta x/\Delta t$', linewidth='10')
plt.plot(tt14cg,max14cg,label=r'$CG4/2\Delta x/\Delta t$', linewidth='6')
plt.plot(tt110multi,max110multi,'--',label=r'$CG2/\frac{\Delta x}{2}/\Delta t$', linewidth='4')
plt.plot(tt1200ndt,max1200ndt,'--',label=r'$CG2/\Delta x/\frac{\Delta t}{2}$', linewidth='4')


plt.xlabel(' $t (s)$ ',size=16)

plt.legend(fontsize="16")
plt.ylabel( r'$\max(\eta) (m)$ ',size=16)
plt.grid()

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
plt.savefig('max_TC5.eps')
plt.show(block=True)

plt.gcf().clear()
