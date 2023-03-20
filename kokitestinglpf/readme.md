# Testing ground for (linear) potential flow (from variational principle -VP)

20-03-03 Latest waveflap code (sorry should go in other folder); case 233 waveflap. Choice of energy unclear; halving dt does not 1/4 energy after t>=28Tp. To do: (a) check whether it does for piston case; (b) compare piston case 233 and case 23; c) compare b) with Yang's code for nz=1 (and other nz) choice;  (d) fix energy issue; (e) fix why using g(z-H0) does not yield same result as integrated/manipulated version at xi3=H0 and xi=0.

16-03-2023, use:
param_psi    = {'ksp_type': 'preonly', 'pc_type': 'lu', 'pc_factor_mat_solver_type':'mumps','snes_monitor':None, 'ksp_monitor':None}

25-02 Get weights of GLL points (by Koki on Slack):
```Python
from FIAT.reference_element import UFCInterval
from FIAT.quadrature import GaussLobattoLegendreQuadratureLineRule

n = 3
fiat_rule = GaussLobattoLegendreQuadratureLineRule(UFCInterval(), n)
print(fiat_rule.get_points())
print(fiat_rule.get_weights())

Order of quadrature in assemble (to get rid of warning):
orders = [1, 3]  # horizontal and vertical
quad_rules = []
for order in orders:
    fiat_rule = GaussLegendreQuadratureLineRule(UFCInterval(), order)
    # Check:
    # print(fiat_rule.get_points())
    # print(fiat_rule.get_weights())
    point_set = GaussLegendrePointSet(fiat_rule.get_points())
    quad_rule = QuadratureRule(point_set, fiat_rule.get_weights())
    quad_rules.append(quad_rule)
quad_rule = TensorProductQuadratureRule(quad_rules)

mesh = UnitIntervalMesh(2)
extm = ExtrudedMesh(mesh, 4)
V = FunctionSpace(extm, "CG", 2, vfamily="CG", vdegree=6)
x, z = SpatialCoordinate(extm)
f = Function(V).interpolate(x**2 * z**6)
# reduced integration
print(assemble(f * dx(scheme=quad_rule)))
```

24-02: GLL now used for CG elements: see https://link.springer.com/article/10.1007/s11831-019-09385-1

Problme solved, used (thanks):
```Python
phihat = fd.product( (x[1]-(H0/nCGvert)*(nCGvert+1-kk))/(H0-(H0/nCGvert)*(nCGvert+1-kk)) for kk in range(2,nCGvert+1,1) )
or phihat = x[1]/H0
```
and phihat.dx(1) to get its derivative.
https://www.mdpi.com/2297-8747/27/4/63

22-02-2023: Dated update of the code; I am trying to put in a split $\phi(x,z,)=\psi(x,t) \hat{\phi}(z) + \varphi(x,z,t)$ for $\hat{\phi}(z)\ne 1$ but with $\hat{\phi}(z=H_0)=1$. See lines 617-629 for a failed attempt. CG1, CG2, CG3 work. 

18-02-2023: Put dated copy of code here; case 233 can be run in nowaveflap=0 (piston limit) or nowaveflap=1 mode (waveflap); both seem to work. To do: CG2-= to CG4; $f(z)\ne 1$ function but with $f(H_0)=1$ at top.

## Linear potential flow without VP

08-02-2023:
- a) 3D PF code (modified midpoint or SV in time)
- b) add vertical structure function
- c) use one element and Chebychev polynomials
- use for SP2 and SP3.
Fingers crossed a,b,c) 1st set up done late February.

16-01-2023: SV and midpoint seem to work. Working on wave-buoy case; useful https://github.com/mm13jmb/waveEnergyDevice
To do:
- Add split with $\phi(x,z,t)=\psi(x,t) f(z)+\varphi(x,z,t)$ with suitable $f(z)$ st $f(H_0)=1$.
- wavemaker case midpoint.
- 3D case with topography
- energy expressions may need clean up for t=0?

07-01: 21:31 wavemaker added; Please check. Case 21. Bit neater way (than in AR's swe codes) of adding time-dependent wavemaker terms, followed in-code indicated FD-example.

07-01 Nonlinear might work; check it. Might not have enough resolution.

06-01: 21:44pm visual convergence posted latest code. check settings a) and b).
Afternoon: 1st attempt (thx Koki and CC) ...ee.py with extruded mesh and only two cases.

04-01: At Dartington Hall. 
Koki: For query below, use extruded mesh to define a tensor space CGxR space s.t. $\tilde{\phi}$ is uniform in $z$. Then $\varphi$ is CGxCG.

Thanks to Koki, case=111 seems to work; case 2 works but is incorrect; phii has a vertical structure (turn facc=0,1 on or off and compare). File ..svpff.py uploaded. Another question is whether the VP with only $\phi(x,z,t)$ and $\eta(x,t)$ can work, rather than having to split into $\phi=\tilde{\phi}(x,t)+\varphi(x,z,t)$.

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

```from firedrake import *
 

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
print(u.dat.data[1])```

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
-\Bigl[\frac12 \frac{L_{w}^2}{W}h (\phi_{x} + (z/h) h_{x})\phi_{z})^2  + \frac12 W \frac{H_{0}^2}{h} (\phi_{z})^2 \Bigr] d z d x$$

$$ + \int_0^{L_x}  -g H_0 W h (\frac 12 h-H_0) + H_0 \phi|_{z=H_0} (W h_{t}-(x-L_w)R_t h_x ) d x - \int_0^{H_0} {L_w R_t \phi h} |_{x=0} dz  dt$$

Time-discrete version SE (not sure if I have done this correctly on $W$ if $W\phi$ are seen as conjugate pair):

$$ 0 = \delta \int_{0}^{L_x} \int_{0}^{H_{0}}
-\Bigl[\frac12 \frac{L_{w}^2}{W^{n+1/2}}h^n (\phi^{n+1}_{x} + (z/h^n) h^n_{x})\phi^{n+1}_{z})^2  + \frac12 W^{n+1/2} \frac{H_{0}^2}{h^n} (\phi^{n+1}_{z})^2 \Bigr] d z d x $$

$$+\int_0^{L_x}  -g H_0 W^{n+1/2} h^n (\frac 12 h^n-H_0) + H_0 W^{n+1}  \phi^{n+1}|_{z=H_{0}} \frac {(h^{n+1}-h^n)}{\Delta t}  + H_0 W^{n}  \phi^{n}|_{z=H_{0}} \frac {h^n}{\Delta t}  -H_0 \phi^{n+1}|_{z=H_0}(x-L_w) R_t^{n+1/2} h_x^{n}  d x$$

$$ - \int_0^{H_0}  {L_w R^{n+1/2}_t \phi^{n+1} h^n} |_{x=0}  d z dt$$

Sinc $W, R$ and $\phi$ as much as possble so at $n+1$ except in one time-step term (calcuate and monitor Kamiltonian):

$$ 0 = \delta \int_{0}^{L_x} \int_{0}^{H_{0}}
-\Bigl[\frac12 \frac{L_{w}^2}{W^{n+1}}h^n (\phi^{n+1}_{x} + (z/h^n) h^n_{x})\phi^{n+1}_{z})^2  + \frac12 W^{n+1} \frac{H_{0}^2}{h^n} (\phi^{n+1}_{z})^2 \Bigr] d z d x $$

$$+\int_0^{L_x}  -g H_0 W^{n+1} h^n (\frac 12 h^n-H_0) + H_0 W^{n+1}  \phi^{n+1}|_{z=H_{0}} \frac {(h^{n+1}-h^n)}{\Delta t}  + H_0 W^{n}  \phi^{n}|_{z=H_{0}} \frac {h^n}{\Delta t}  -H_0 \phi^{n+1}|_{z=H_0}(x-L_w) R_t^{n+1} h_x^{n}  d x$$

$$ - \int_0^{H_0}  {L_w R^{n+1}_t \phi^{n+1} h^n} |_{x=0}  d z dt$$



Time-discrete version mid-point:

$$ 0 = \delta  \int_{0}^{L_x} \int_{0}^{H_{0}}
-\Bigl[\frac12 \frac{L_{w}^2}{W^{n+1/2}}h^{n+1/2} (\phi^{n+1/2}_{x} + (z/h^{n+1/2}) h^{n+1/2}_{x})\phi^{n+1/2}_{z})^2  + \frac 12 W^{n+1/2} \frac{H_{0}^2}{h} (\phi^{n+1/2}_{z})^2 \Bigr] d z d x $$

$$ + \int_0^{L_x}  -g H_0 W^{n+1/2} h^{n+1/2} (\frac 12 h^{n+1/2}-H_0) -H_0 \phi^{n+1/2}(x-L_w)R^{n+1/2}_t h^{n+1/2}_x d x $$

$$+ \int_0^{H_0} H_0 W^{n+1/2} \phi^{n+1/2}  \frac {(h^{n+1} -h^n)}{\Delta t}- H_0 h^{n+1/2}\frac {(W^{n+1} \phi^{n+1}-W^n \phi^n)}{\Delta t}  - H_0 W^{n+1} \phi^{n+1}  \frac {(h^{n+1}-h^n)}{\Delta t} + H_0 W^n \phi^n  \frac {h^n}{\Delta t} d x $$

$$-\int_0^{H_0} {L_w R^{n+1/2}_t \phi^{n+1/2} h^{n+1/2}} |_{x=0} dz  $$

Variations wrt $\h^{n+1/2},\phi^{n+1/2}|_{z=H_0},\phi^{n+1/2}$ in interior or $\phi=\phi|_{z=H_0}+\varphi$ with latter $0$ at $z=H_0$.

Step-1: Variational derivative wrt $h^n$. Nonlinear solve for $\phi^{n+1}$ at $z=H_0$ free surface.

Step=2: Variational derivative wrt $\phi^{n+1}$ but in interior with given new solution $\phi^{n+1}$ at $z=H_0$ free surface; linear solve.

Bad news is that Steps 1 and 2 need to be solved in tandem. Testing that in linear system will likely already be an ordeal given the way Firedrake's bc's are set up. Koki, David, Colin? That is why we had integrated out the vertical 5 years ago. Aye.

Step=3: Variational derivative wrt $\phi^{n+1}$ but at free surface $z=H_0$ to solve for $h^{n+1}$; linear solve.

19-12: nonlinear solvers not yet resolved.
