# Testing ground for (linear) potential flow (from variatonal principle --VP)

## Linear potential flow without VP

06-12-2022 Onno: Clean-up of code, made scaling simple and redid exact solution.

code: potflowwdsvp.py

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


