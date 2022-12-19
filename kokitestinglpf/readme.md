# Testing ground for (linear) potential flow (from variational principle -VP)

## Linear potential flow without VP

19-12: Coded up VP nonlinear case (nvpcase = 2); can be run with fac=0 such that nonlinearity switched off; that works; poor or failed convergence for fac=1.0 (nonlinear case); not sure which solvers to use. Last two steps are linear: how can that be enforced? How does one choose the solvers?
 

18-12 Standard weak approach (nvpcase = 0) and VP approach (nvpcase = 1) seem to both work and give same result; please check; but comparison with exact solution seems off? Use: potflowwdsvpf.py
Just copied file here; not sure yet how commit works. 
- To do: CG2 does not work; output function complains? Why?
- To do: check with exact solution --found below? Needs sorting as check!
- To do: make faster.

- To do: add wavemaker
- To do: work out nonlinear case and implement.

15-12: energy plotted; seems to behave like SE oscillations halving with dt-> 0.5*dt; exact and numerical solutions plotted together; terribly slow? Why?
- To do: faster, check whether code can converges a bit better; steps 2 and 3 are linear also for nonlinear model so linear fast solvers are possible.
Attempts do not work yet.
- To do: CG2 does not work; output function complains? Why?
- To do: VP version as discussed; discussion with Koki found als in the code as comments. Seem to work now see 18-12.

09-12: performance still very poor relative to exact solution at nx=120 and nz=6 and 2000 time steps in one period?

08-12: probably got it for nomrla part: had made silly mistake by assigning incorrectly before end of time loop so no update was made.

06-12-2022 Onno: Clean-up of code, made scaling simple and redid exact solution.

code: potflowwdsvp.py
See comments 07-12-2022
Something is terribly wrong in the code; I have no contact/link with exact solution.

However, for the runtime case nvpcase == 0 the plots should show a wave for $\eta$ in its four phases at $0, \pi/2, \pi, 3\pi/2 and 2\pi$, i.e. at $t=(0,1/4,1/2,3/4,1) T_{period}$. So if $\eta$ is 0 at the start it should be zero at $(0,1/2,1)T_{period}$ and at the moment movement seems to be very small as if time increments are wrong. Energy needs to be checked, as monitoring.

## Linear solution used in code

Solving linear potential flow:

$\nabla^2 \phi=0\quad[x,z]\in[0,L_x]\times[0,H_0]$ for $\phi(x,z,t)$.

$\partial_t\phi+g \eta = 0$ on $x\in[0,L_x]$ and $z=H_0$ for $\phi(x,z=H_0,t),\eta(x,t)$.

$\partial_t \eta = \partial_z\phi$ on $x\in[0,L_x]$ and $z=H_0$ for $\phi(x,z=H_0,t)$.

Neumann conditions at $z=0,x=0,L_x$.

Solutions:

$\phi(x,z,t) = D \cos(k x)\cosh(k z)\cos(\omega t)$ or $\phi(x,z,t) = D \cos(k x)\cosh(k z)\sin(\omega t)$ 

$\eta(x,t) = A \cos(k x) \sin(\omega t)$ or $\eta(x,t) = A \cos(k x) \cos(\omega t)$

with $D = g A/(\omega\cosh(k H_0))$ or $D = -g A/(\omega\cosh(k H_0))$ and $\omega = \sqrt{g k \tanh(k H_0)}$.

## Nonlinear flow VP

No wavemaker, topography $b=0, \partial_y =0$:

$$ 0 = \delta \int_0^T \int_{0}^{L_x} \int_{0}^{H_{0}}
\Bigl[\frac12 \frac{L_{w}^2}{W}h (\phi_{x} + (z/h) h_{x})\phi_{z})^2  + \frac12 W \frac{H_{0}^2}{h} (\phi_{z})^2 \Bigr] d z d x $$

$$  +\int_0^{L_x} H_{0} \Bigl( g W h (\frac 12 h-H_0) - \phi W h_{t} \Bigr)_{z=H_{0}} d x dt $$
