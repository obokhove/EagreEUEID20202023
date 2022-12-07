# Testing ground for (linear) potential flow (from variatonal principle --VP)

## Linear potential flow without VP

06-12-2022 Onno: Clean-up of code, made scaling simple and redid exact solution.

code: potflowwdsvp.py

However, for nvpcase == 0 the plots should show a wave for eta in its 5 phases at 0, pi/2, pi, 3pi/2 and 2*pi, i.e.
at t=(0,1/4,1/2,3/4,1)*Tperiod.
So if eta is 0 bto start it should be zero at (0,1/2,1)*Tperiod and at the moment movement seems to be very small as if time is wrong.
Energy needs to be checked, as monitor.


