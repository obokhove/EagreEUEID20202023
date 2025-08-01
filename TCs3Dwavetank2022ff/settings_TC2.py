# new 2D TC2 based on the same problem considered in TC1

from firedrake import *
import numpy as np

"""
    *********************************************
    *                 Test case                 *
    *********************************************"""
def test_case():
    #________________ Kind of data ________________#
    #input_data = "measurements"  # from experiments
    input_data = "created"       # set the wavemaker
    #______________ Temporal scheme _______________#
    scheme = "MMP"
    # "SE": Symplectic-Euler ; "SV": Stormer-Verlet
    # "MMP": use the Modified Mid-Point VP approach
    #__________________ Dimension _________________#
    dim = "2D"
    #"2D": R(t) and b(x); "3D": R(y,t) and/or b(x,y)
    #______ Path and name of the saved files ______#
    # old: save_path = 'data/'+scheme+'/TC2_series/Nx800_nz6_2dtf/'
    # 
    # NB: Look at the post-processing files to set save_path
    # 
    save_path = 'data/'+scheme+'/TC2_series/Nx800_nz6_2dtf/'
    # ----yl added. whether the seabed is flat or not
    bottom = 'flat' 
    # 'flat':b(x,y)=0; 'nonuniform':b(x,y)!=0
    # ----yl added. Whether or not to apply mild-slope approximation (MSA)
    FWF = 0
    # 1: use full weak forms (FWF); 0: use mild-slope approximations.
    save_pvd = False
    # Whether or not to save the 3D results into pvd files 
    return input_data, scheme, dim, save_path, bottom, FWF, save_pvd

"""
    *********************************************************************
    *                         Numerical domain                          *
    *********************************************************************"""
def domain(bottom):
    #______________________ Beach ______________________#
    m = 2
    Lx = 2*pi
    k = 2*pi*m/Lx
    g  = 9.81
    kH0 = np.arccosh(0.5*g)
    H0 = kH0/k
    xb = 0.0                                          # Start of the beach
    sb = 0.0                                           # Slope of the beach
    # yl update
    def H_expr(function,x):
        function.interpolate(H0-conditional(le(x[0],xb),0.0,sb*(x[0]-xb)))
    #______________________ Basin ______________________#
    Hend = 0.5                              # Depth at the end of the beach
    if bottom=='nonuniform':
        Lx = xb +(H0-Hend)/sb                                 # Length in x
    else:
        Lx = 2*pi
    Ly = 1.0                                                  # Length in y
    Lw = 1.0                                        # End of the x-transform
    res_x = 2*pi/800                                            # x-resolution
    res_y = 1.0                                            # y-resolution
    n_z = 6                                         # Order of the expansion
    return k, H0, xb, sb, H_expr, Hend, Lx, Ly, Lw, res_x, res_y, n_z


"""
    **************************************************************************
    *                                Wavemaker                               *
    **************************************************************************"""
def wavemaker(k, dim, H0, Ly, Lw, input_data):
    #_____________________________ Characteristics _____________________________#
    g = 9.81                                             # Gravitational constant
    w = sqrt(2*k*np.sinh(k*H0)) 
    Tw = 2*pi/w                                                     # Wave period
    gamma = 0.0                                                  # Wave amplitude
    t_stop = Constant(3*Tw)                                     
    Aa = 0.01
    Bb = 0.01

    # yl update
    if input_data=='created':
        if dim == "2D":
            def WM_expr(function,x,t,t_stop):
                if t.values()[0] <= t_stop.values()[0]:
                    function.interpolate(conditional(le(x[0],Lw),-gamma*cos(w*t),0.0))
                else:
                    function.interpolate(conditional(le(x[0],Lw),-gamma*cos(w*t_stop),0.0))

            def dWM_dt_expr(function,x,t,t_stop):
                if t.values()[0] <= t_stop.values()[0]:
                    function.interpolate(conditional(le(x[0],Lw),gamma*w*sin(w*t),0.0))
                else:
                    function.assign(0.0)

            def dWM_dy_expr(function,x,t,t_stop):
                function.assign(0.0)

        elif dim == "3D":
            def WM_expr(function,x,t,t_stop):
                if t.values()[0] <= t_stop.values()[0]:
                    function.interpolate(conditional(le(x[0],Lw), gamma*((x[1]-0.5*Ly)/(0.5*Ly))*cos(w*t),0.0))
                else:
                    function.interpolate(conditional(le(x[0],Lw), gamma*((x[1]-0.5*Ly)/(0.5*Ly))*cos(w*t_stop),0.0))
            
            def dWM_dt_expr(function,x,t,t_stop):
                if t.values()[0] <= t_stop.values()[0]:
                    function.interpolate(conditional(le(x[0],Lw),-gamma*w*((x[1]-0.5*Ly)/(0.5*Ly))*sin(w*t),0.0))
                else:
                    function.assign(0.0)

            def dWM_dy_expr(function,x,t,t_stop):
                if t.values()[0] <= t_stop.values()[0]:
                    function.interpolate(conditional(le(x[0],Lw), gamma*cos(w*t)/(0.5*Ly),0.0))
                else:
                    function.interpolate(conditional(le(x[0],Lw), gamma*cos(w*t_stop)/(0.5*Ly),0.0))

    elif input_data == "measurements": # linear interpolation of experimental data
        def WM_expr(function, x, Rt):  # Rt is a Constant
            function.interpolate(conditional(le(x[0],Lw),Rt,0.0))
        def dWM_dt_expr(function, x, Rt_t): # Rt_t is a Constant
            function.interpolate(conditional(le(x[0],Lw),Rt_t,0.0))
        def dWM_dy_expr(function):
            function = Constant(0.0)

    def h_ex_expr(function,x,t):
        function.interpolate( H0 + cos(k*x[0])*(Aa*cos(w*t)+Bb*sin(w*t)) )

    def phis_ex_expr(function,x,t):
        function.interpolate( cos(k*x[0])*(exp(k*H0)+exp(-k*H0))*(-Aa*sin(w*t)+Bb*cos(w*t))/w )

    def phii_ex_expr(function,x,t):
        if dim=='3D':
            function.interpolate( cos(k*x[0])*(exp(k*x[2])+exp(-k*x[2]))*(-Aa*sin(w*t)+Bb*cos(w*t))/w )
        elif dim=='2D':
            function.interpolate( cos(k*x[0])*(exp(k*x[1])+exp(-k*x[1]))*(-Aa*sin(w*t)+Bb*cos(w*t))/w )

    return g, w, Tw, gamma, t_stop, Aa, Bb, WM_expr, dWM_dt_expr, dWM_dy_expr, h_ex_expr, phis_ex_expr, phii_ex_expr


"""
    ***********************************
    *               Time              *
    ***********************************"""
def set_time(Lx, res_x, Ly, res_y, H0, n_z, Tw):
    Nx = round(Lx/res_x)    # Number of elements in x (round to the nearest integer)
    Ny = round(Ly/res_y)    # Number of elements in y
    dz = H0/n_z
    dx = Lx/Nx
    dt = ((Lx/3200)/(2*pi))*2        # time step
    T0 = 0.0                     # Initial time
    t = T0                  # Temporal variable 
    dt_save = dt*32          # saving time step
    Tend = 3*Tw                    # Final time
    return T0, t, dt, Tend, dt_save
