import os.path
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

fileE = 'resultsntflag0/potflow3dperenergy.txt'
outputE = open(fileE,'r')
#  infile.readline() # skip the first line not first line here (yet)
tijd = []
EPot = []
EKin = []
Etot = []
for line in outputE:
    words = line.split()
    # words[0]: tijd, words[1]: relEnergy
    tijd.append(float(words[0]))
    EPot.append(float(words[1]))
    EKin.append(float(words[2]))
    Etot.append(float(words[3]))

outputE.close()
plt.figure(101)
plt.plot(tijd,EPot,'--b')
plt.plot(tijd,EKin,'-.r')
plt.plot(tijd,Etot,'-k')
plt.xlabel(' $t (s)$ ',size=16)
plt.ylabel(' $E_{kin}, E_{pot}, E_{tot}$ ',size=16)
# plt.axes([0,10,0.001,0.002])
#plt.yticks([0.0010082,0.0010083,0.0010084,0.0010085,0.0010086])
# plt.axes(xlim=(0, 10), ylim=(0.0010082, 0.0010086), autoscale_on=False) # [0 10 0.0010082 0.0010086])

plt.show(block=True)
plt.pause(0.001)
plt.gcf().clear()
plt.show(block=False)

print("Finished program!")
