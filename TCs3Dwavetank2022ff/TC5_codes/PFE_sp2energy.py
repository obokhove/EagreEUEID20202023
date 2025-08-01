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
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt200ndt=np.array(tijd)
max200ndt=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()


fileE = 'C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/delt10-5nt100/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt=np.array(tijd)
maxx=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()


fileE = 'C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/delt10-5muti10/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt10multi=np.array(tijd)
max10multi=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

fileE = 'C:/Users/Junho/Downloads/two_interactingPmat.py/data/H2/delt10-5nt100cg4/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
# EPot = []
# EKin = []
max_data = []
AA_data=[]
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    # EPot.append(float(words[]))
    # EKin.append(float(words[2]))
    max_data.append(float(words[3]))
    # AA_data.append(float(words[2]))


tt4cg=np.array(tijd)
max4cg=np.array(max_data)
# AA3=np.array(AA_data)*(4/3)**(1/3)
outputE.close()

E0=max_data[1,0]+max_data[2,0]



plt.figure(figsize=(10,6))
plt.plot(tt,abs(maxx/E0),label=r'$CG2/\Delta x/\Delta t$', linewidth='4')
plt.plot(tt4cg,abs(max4cg/E0),label=r'$CG4/2\Delta x/\Delta t$', linewidth='4')
plt.plot(tt10multi,abs(max10multi/E0),label=r'$CG2/\frac{\Delta x}{2}/\Delta t$', linewidth='4')
plt.plot(tt200ndt,abs(max200ndt/E0),label=r'$CG2/\Delta x/\frac{\Delta t}{2}$', linewidth='4')





plt.xlabel(' $t (s)$ ',size=16)

plt.legend(fontsize="16")
plt.ylabel( r'$|(E(t)-E(0))/E(0)|$ ',size=16)
plt.grid()

plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) # [0 10 0.0010082 0.0010086])
plt.savefig('energy_TC5.eps')
plt.show(block=True)

plt.gcf().clear()

