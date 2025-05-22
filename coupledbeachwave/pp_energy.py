# Post-processing code for plotting the energy evolution 
# and water line x_c of the 2D coupled wavetank.

import numpy as np
import matplotlib.pyplot as plt
import os.path

#------ User Input ------#
test_case = "FG_FWF_8April"

H0 = 1.0
g = 9.81                                             
lamb = 2.0         # Wavelength
k = 2*np.pi/lamb   # Wave number
w = np.sqrt(g*k*np.tanh(k*H0))   # Wave frequency
Tw = 2*np.pi/w     # Wave period
t_stop = 60*Tw     # When to stop the wavemaker

comparison=False
test_case2 = "2DTC3_xc8_nz8_tconst"

save_figure=False
figure_name='energy_and_xw.png'
#------------------------#
data_path = 'data/' + test_case
file = os.path.join(data_path,'energy.csv')

with open(file,'r') as f:
    time, E_dw, E_sw, s_w = np.loadtxt(f, usecols=(0,1,2,3), unpack=True)

E_tot = E_dw + E_sw

if save_figure:
    save_path=os.path.join(data_path, figure_name)

if comparison:
    data_path2 = 'data/' + test_case2
    file2 = os.path.join(data_path2,'energy.csv')
    with open(file2,'r') as f2:
        time2, E_dw2, E_sw2, s_w2 = np.loadtxt(f2, usecols=(0,1,2,3), unpack=True)
        
# energy absorbing
t1_0 = 55 * Tw
t1_1 = 60 * Tw
mask1 = (time >= t1_0) & (time <= t1_1)
avg1 = np.mean(E_tot[mask1])
print(f"Average E_tot between {t1_0}s and {t1_1}s: {avg1}")

t2_0 = 105 * Tw
t2_1 = 110 * Tw
mask2 = (time >= t2_0) & (time <= t2_1)
avg2 = np.mean(E_tot[mask2])
print(f"Average E_tot between {t2_0}s and {t2_1}s: {avg2}")

print((avg1-avg2)/avg1)

fig, [ax1, ax2] = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [4, 3]})
fig.set_size_inches(10,9)
fig.set_tight_layout(True)

ax1.set_title('Energy variations of the coupled wavetank', fontsize=18)
#ax1.set_xlabel('$t$ [s]',fontsize=14)
ax1.set_ylabel('$E$ [J]',fontsize=14)
ax1.plot(time, E_dw, 'b-', label="Deep water")
ax1.plot(time, E_sw, 'r-', label="Shallow water")
ax1.plot(time, E_tot,'y-', label="Whole domain")
ax1.set_xlim(time[0], time[-1])

ax2.set_title('Evolution of the water line', fontsize=18)
ax2.set_xlabel('$t$ [s]',fontsize=14)
ax2.set_ylabel('$x_W$ [m]',fontsize=14)
ax2.plot(time, s_w,'c-', label="Water Line")

# Indicate the time when the wavemaker stops
ax1.axvline(x=t_stop, color='green', linestyle=':', label='Wavemaker switched off')

if comparison:
    ax1.plot(time2, E_dw2, 'bo', label="tconst", markersize=5)
    ax1.plot(time2, E_sw2, 'ro', label="tconst", markersize=5)
    ax2.plot(time2, s_w2, 'go', label="Water Line", markersize=5)

ax1.legend(loc='upper right',fontsize=14)  
ax1.grid()
#ax2.legend(loc='upper left',fontsize=14)  
ax2.grid()

if save_figure:
    plt.savefig(save_path,dpi=300)
else:
    plt.show()