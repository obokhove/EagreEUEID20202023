## Potential-flow dynamics 3D VP-based

22-03-2022 Partially fixed by correcting name phi_f into psi_f. However, corrsp2 still does not work. The SP2 paraview movie blows up 10^43. However the yslice's (at y=0, Ly/2 and Ly) seem fine and periodic. No idea what is going on. Note that time loop switched off. See start of while loop.
Newer file added: can't figure out why psi_f does not show normally in paraview; same for U0y, c0y and the combination. I had it at some point but something has changed.

20-03-03 Latest waveflap code (sorry should go in other folder); case 233 waveflap. Choice of energy unclear; halving dt does not 1/4 energy after t>=28Tp. To do: (a) check whether it does for piston case; (b) compare piston case 233 and case 23; c) compare b) with Yang's code for nz=1 (and other nz) choice;  (d) fix energy issue; (e) fix why using g(z-H0) does not yield same result as integrated/manipulated version at xi3=H0 and xi=0.
 
07-03-2023 MMP and SV via VP now both work (by Onno with ".solve" and "psi_f", use psisv, EPot corrections by Junho). Note that I have cleaned up the code a bit more as well. Parameter settings need to be improved. See Appendix G in pdf-pp1718.  Also added the tic-tic timing around the time loop ("import time as tijd").

Upon making the splitting $\phi(x,y,z,t)=\psi(x,y,t)\hat{\phi}(z)+\varphi(x,y,z,t)$ with $\hat{\phi}(H_0=1)$ and $\varphi(x,y,H_0,t)=0$ (or another splitting with $z$-dependence), the MMP time discretisation reads

$$ 0=  \int_{\hat{\Omega}_{x,y}}\bigg[ \Big( -H_0 W\psi^{n+1/2}\frac {(h^{n+1}-h^n)}{\Delta t}  + H_0 h^{n+1/2}\frac {(W\psi^{n+1}-W\psi^n)}{\Delta t} +H_0 g W h^{n+1/2}(\dfrac12 h^{n+1/2} -H_0) \Big)$$

$$+\int_0^{H_0} \bigg[\dfrac12 \dfrac{L_w^2}{W}h^{n+1/2} \big(\psi^{n+1/2}_x\hat{\phi}+\varphi^{n+1/2}_x- \dfrac{1}{h^{n+1/2}}\big(H_0 {b_x}+z h^{n+1/2}_x\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \big)^2$$

$$ +\frac12 W{h^{n+1/2}}\bigg( \psi^{n+1/2}_y\hat{\phi}+\varphi^{n+1/2}_y -\dfrac{1}{h^{n+1/2}}\big(H_0 {b_y}+z h^{n+1/2}_y\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \bigg)^2$$

$$ +\dfrac12W\dfrac{H_0^2}{h^{n+1/2}}(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z)^2 \bigg] {\rm d}z \bigg]{\rm d}x {\rm d}y.$$

The time-discrete VP corresponding to SV reads

$$ 0=  \int_{\hat{\Omega}_{x,y}}\bigg[
\Big( -H_0 W\psi^{n+1/2}\frac {(h^{n+1}-h^n)}{\Delta t} + H_0 W\psi^{n+1}\frac {h^{n+1}}{\Delta t} - H_0 W\psi^{n}\frac {h^{n}}{\Delta t} +\frac 12 H_0 g W \bigl(
h^{n+1}(\dfrac12 h^{n+1} -H_0)+ h^{n}(\dfrac12 h^{n} -H_0)\bigr)\Big)$$

$$+\frac 12 \int_0^{H_0} \bigg[\dfrac12 \dfrac{L_w^2}{W}h^{n+1}
\big(\psi^{n+1/2}_x\hat{\phi}+\varphi^{n+1/2}_x- \dfrac{1}{h^{n+1}}\big(H_0 {b_x}+z h^{n+1}_x\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \big)^2$$

$$ +\frac12 W{h^{n+1}}\bigg(
\psi^{n+1/2}_y\hat{\phi}+\varphi^{n+1/2}_y -\dfrac{1}{h^{n+1}}\big(H_0 {b_y}+z h^{n+1}_y\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \bigg)^2$$

$$ +\dfrac12W\dfrac{H_0^2}{h^{n+1}}(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z)^2 \bigg] {\rm d}z$$

$$ +\frac 12 \int_0^{H_0} \bigg[\dfrac12 \dfrac{L_w^2}{W}h^{n}
\big(\psi^{n+1/2}_x\hat{\phi}+\varphi^{n+1/2}_x- \dfrac{1}{h^{n}}\big(H_0 {b_x}+z h^{n}_x\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \big)^2$$

$$ +\frac12 W{h^{n}}\bigg(
\psi^{n+1/2}_y\hat{\phi}+\varphi^{n+1/2}_y -\dfrac{1}{h^{n}}\big(H_0 {b_y}+z h^{n}_y\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \bigg)^2$$

$$+\dfrac12W\dfrac{H_0^2}{h^n}(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z)^2 \bigg]{\rm d}z
\bigg]{\rm d}x {\rm d}y.$$

References on the (fd.)derivative command and such: https://fenicsproject.org/pub/book/book/fenics-book-2011-06-14.pdf (section 17.5.1) https://arxiv.org/abs/1211.4047 (section 6.4).

## SP1 test: one soliton

To do: write down (dimensional) expressions. Strategy (since what I have seen is incomprehensible): use one for BL-system (dimensionless) and then scale back to dimensional case. Done. Tests seem to work.

## SP2 test: two interacting solitons

To do: write down (dimensional) expressions.

## SP3 test: three interacting solitons

To do: write down (dimensional) expressions.


