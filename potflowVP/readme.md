## Potential-flow dynmamics 3D VP-based

07-03-2023 MMP and SV via VP now both work (by Onno with ".solve" and "psi_f", EPot corrections by Junho). Note that I have cleaned up the code a bit more as well. Parameter settings need to be improved. See Appendix G in pdf-pp1718. 

Upon making the splitting $\phi(x,y,z,t)=\psi(x,y,t)\hat{\phi}(z)+\varphi(x,y,z,t)$ with $\hat{\phi}(H_0=1)$ and $\varphi(x,y,H_0,t)=0$, the MMP time discretisation o reads

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

$$ +\dfrac12W\dfrac{H_0^2}{h^{n+1}}(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z)^2 \bigg]\,{\rm d}z$$

$$ +\frac 12 \int_0^{H_0} \bigg[\dfrac12 \dfrac{L_w^2}{W}h^{n}
\big(\psi^{n+1/2}_x\hat{\phi}+\varphi^{n+1/2}_x- \dfrac{1}{h^{n}}\big(H_0 {b_x}+z h^{n}_x\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \big)^2$$

$$ +\frac12 W{h^{n}}\bigg(
\psi^{n+1/2}_y\hat{\phi}+\varphi^{n+1/2}_y -\dfrac{1}{h^{n}}\big(H_0 {b_y}+z h^{n}_y\big)(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z) \bigg)^2$$

$$+\dfrac12W\dfrac{H_0^2}{h^n}(\psi^{n+1/2}\hat{\phi}_z+\varphi^{n+1/2}_z)^2 \bigg]{\rm d}z
\bigg]{\rm d}x {\rm d}y.$$

