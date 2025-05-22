# Settings file for the 2D coupled wavetank
# Main file: "coupled_tank.py"
# Adapted from the settings file for the 3D wavetank with only 2D features kept

from firedrake import *

"""
    *********************************************
    *                 Test case                 *
    *********************************************"""
def test_case():
    #________________ Kind of data ________________#
    #input_data = "measurements"  # from experiments
    input_data = "created"       # set the wavemaker
    #________ folder name of the test case ________#
    test_case = "FG_FWF_8April"
    # yl added. Whether or not to apply mild-slope approximation (MSA)
    FWF = 1 # 1: use full weak forms (FWF); 0: apply MSA
    # yl added. Whether or not to save the 3D results into pvd files
    save_pvd = True # not actually used in the main code
    return input_data, test_case, FWF, save_pvd

"""
    *********************************************************************
    *                         Numerical domain                          *
    *********************************************************************"""
def domain():
    #____________________ Whole Domain ______________________#
    H0 = 1.0                       # Depth at rest (flat bottom)
    xb = 3.0                       # Start of the beach
    sb = 0.1                       # Slope of the beach
    
    #______________________ Deep Water ______________________#
    Hc = 0.2                                               # Depth at x_c
    Lx = xb +(H0-Hc)/sb            # Length of the deep-water region, x_c
    Lw = 1.0                                       # End of the x-transform
    res_x = 0.05                                    # deep-water resolution
    n_z = 8                                        # Order of the expansion
    return H0, xb, sb, Hc, Lx, Lw, res_x, n_z


"""
    **************************************************************************
    *                                Wavemaker                               *
    **************************************************************************"""
def wavemaker(H0, Lw):
    #_____________________________ Characteristics _____________________________#
    g = 9.81                                             # Gravitational constant
    lamb = 2.0                                                       # Wavelength
    k = 2*pi/lamb                                                   # Wave number
    w = sqrt(g*k*tanh(k*H0))                                     # Wave frequency
    Tw = 2*pi/w                                                     # Wave period
    gamma = 0.02                                           # Wave maker amplitude
    t_stop = 60*Tw # 60*Tw                                   # When to stop the wavemaker

    return g, lamb, k, w, Tw, gamma, t_stop


"""
    ***********************************
    *               Time              *
    ***********************************"""
def set_time(Tw):
    dt = 0.001                      # Time step 
    dt_save = 0.02           # Saving time step
    T0 = 0.0                     # Initial time
    Tend = 110*Tw # 110*Tw                   # Final time 
    t = T0                  # Temporal variable
    return T0, t, dt, Tend, dt_save
