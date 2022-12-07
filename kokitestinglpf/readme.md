# Testing ground for (linear) potential flow (from variatonal principle --VP)

## Linear potential flow without VP

06-12-2022 Onno: Clean-up of code, made scaling simple and redid exact solution.

code: potflowwdsvp.py

However, for nvpcase == 0 the plots should show a wave for eta in its 5 phases at 0, pi/2, pi, 3pi/2 and 2*pi, i.e.
at t=(0,1/4,1/2,3/4,1)*Tperiod.
So if eta is 0 bto start it should be zero at (0,1/2,1)*Tperiod and at the moment movement seems to be very small as if time is wrong.
Energy needs to be checked, as monitor.

##Linear solution

Solving linear potential flow:

$\nabla^2 \phi=0\quad[x,z]\in[0,L_x]\times[0,H_0]$ for $\phi(x,z,t)$.

$\partial_t\phi+g \eta = 0$ on $x\in[0,L_x]$ and $z=H_0$ for $\phi(x,z=H_0,t),\eta(x,t)$.

$\partial_t \eta = \partial_z\phi$ on $x\in[0,L_x]$ and $z=H_0$ for $\phi(x,z=H_0,t)$.


