# Testing ground for (linear) potential flow (from variational principle -VP)

## Linear potential flow without VP

07-01 Nonlinear might work; check it. Might not have enough resolution.

06-01: 21:44pm visual convergence posted latest code. check settings a) and b).
Afternoon: 1st attempt (thx Koki and CC) ...ee.py with extruded mesh and only two cases.

04-01: At Dartington Hall. 
Koki: For query below, use extruded mesh to define a tensor space CGxR space s.t. $\tilde{\phi}$ is uniform in $z$. Then $\varphi$ is CGxCG.

Thanks to Koki, case=111 seems to work; case 2 works but is incorrect; phii has a vertical structure (turn facc=0,1 on or off and compare). File ..svpff.py uploaded. Another question is whether the VP with only $\phi(x,z,t)$ and $\eta(x,t)$ can work, rather than having to split into $\phi=\tilde{phi}(x,t)+\varphi(x,z,t)$.

01-01: Removed ==0 following remark on slack: incompatible function space again.

30-12: mixed variables stuff including their bc's completely incomprehensible; why is there no clear instruction page?
Put latest copy here as ...copypy and posted on slack. Quite stuck and cannot yet find similar examples. The warning on mixed spaces in the online FD-manual seems to apply but information is too vague for me to be comprehensible.

25/26-12: Quite bizarre that code does not work; in essence, it does the same and is set up similarly as 3D_tank.py at 

 https://github.com/EAGRE-water-wave-impact-modelling/3D-wave-tank-JCP2022

(so Gidel and Lu already have solved a similar nearly the same system, with in a first step phi at surface and (var)phi in interior solved in tandem --it is irritating that there still is a bug).
See solver_full.py routine/function WF_psi_star; follow the lead, mixed variable etc. So where is the error/difference in approach; perhaps taking with Gidel/Lu regularly would help? psi1 and psihat there are solved in tandem.


24-12: Case 111. Don't understand how system phif_expr1=0 and phi_expr1=0 are imposed in tandem; why is phif_expr1+phi_expr1==0 doing so? However, version does not work; ufl.log.UFLException: Not an UFL type: <class 'ufl.equation.Equation'>

23-12: Still stuck despite update; defined on incompatible FunctionSpace! See attached file.

20-12: Case 111 in progress; the du's need to be updated next, I think such that right function space is used, per Koki's remarks (removed comment sign):

from firedrake import *
 

mesh = UnitSquareMesh(1, 1, quadrilateral=True)
V = FunctionSpace(mesh, "CG", 2)
W = V * V
eta = Function(V).interpolate(Constant(1))
phi = Function(V).interpolate(Constant(2))
u = Function(W)
v = TestFunction(W)
u0, u1 = u.split()   just to assign values
u0.interpolate(Constant(100))
u1.interpolate(Constant(200))
u0, u1 = split(u)   Will later solve for u.
v0, v1 = split(v)   These represent blocks.

 First define VP in terms of independent functions, eta and phi.
VP = (u0 * inner(eta, eta) + inner(phi, phi)) * dx
F0 = derivative(VP, eta, du=v0)   use correct du.
F1 = derivative(VP, phi, du=v1)   use correct du.
F1 = replace(F1, {phi: u1})   replace if needed.
F = F0 + F1   contains both u0 and u1.
A = assemble(F)
print(A.dat.data[0])
print(A.dat.data[1])
solve(F==0, u)
print(u.dat.data[0])
print(u.dat.data[1])

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

No wavemaker, topography $b=0, \partial_y =0$, $W=L_w-R(t)$ for case with wavemaker:

$$ 0 = \delta \int_0^T \int_{0}^{L_x} \int_{0}^{H_{0}}
-\Bigl[\frac12 \frac{L_{w}^2}{W}h (\phi_{x} + (z/h) h_{x})\phi_{z})^2  + \frac12 W \frac{H_{0}^2}{h} (\phi_{z})^2 \Bigr] d z d x 
+ \int_0^{L_x}  -g H_0 W h (\frac 12 h-H_0) + H_0 W  \phi|_{z=H_{0}} h_{t} d x 
+ \int_0^{H_0} L_w R_t phi h |_{x=0} dt $$

Time-discrete version:

$$ 0 = \delta \int_{0}^{L_x} \int_{0}^{H_{0}}
-\Bigl[\frac12 \frac{L_{w}^2}{W}h^n (\phi^{n+1}_{x} + (z/h^n) h^n_{x})\phi^{n+1}_{z})^2  + \frac12 W \frac{H_{0}^2}{h^n} (\phi^{n+1}_{z})^2 \Bigr] d z d x 
+\int_0^{L_x}  -g H_0 W h^n (\frac 12 h^n-H_0) + H_0 W  \phi^{n+1}|_{z=H_{0}} \frac {(h^{n+1}-h^n)}{\Delta t}  + H_0 W  \phi^{n}|_{z=H_{0}} \frac {h^n}{\Delta t}  d x$$

Step-1: Variational derivative wrt $h^n$. Nonlinear solve for $\phi^{n+1}$ at $z=H_0$ free surface.

Step=2: Variational derivative wrt $\phi^{n+1}$ but in interior with given new solution $\phi^{n+1}$ at $z=H_0$ free surface; linear solve.

Bad news is that Steps 1 and 2 need to be solved in tandem. Testing that in linear system will likely already be an ordeal given the way Firedrake's bc's are set up. Koki, David, Colin? That is why we had integrated out the vertical 5 years ago. Aye.

Step=3: Variational derivative wrt $\phi^{n+1}$ but at free surface $z=H_0$ to solve for $h^{n+1}$; linear solve.

19-12: nonlinear solvers not yet resolved.
