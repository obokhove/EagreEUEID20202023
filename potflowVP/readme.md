## Potential-flow dynamics 3D VP-based
09-05:
"I compared elapsed time between two settings; first: nCG=CG2, nx=132, ny=480, second: nCG=CG4, nx=132/2, ny=480/2. Please note that they have the same degree of freedom (dof), that is, dof=(nCG*nx)*(nCG*ny+1)=253704 in the code. Thus, I expected both of them to take a similar amount of time to compute under the same settings except nCG, nx, and ny. However, CG4 spent 1.5 times more time than CG2.
Except dof,  Is there anything else which affects speed of computation?"

" I suspect that the most intensive part of your computation is the additive Schwarz method on an extruded hexahedral mesh. Here your patches seem to be columns on each of the faces of the base mesh. As the polynomial degree p is increased, the number of flops to assemble the matrix scales with p^{2d+1}, and the number of operations to factor and solve the patch problems will grow cubically and quadratically on the number of DOFs on each column (if you fix the DOFs, at high order you have fewer columns with more DOFs compared to low-order). Because of the nonlinear dependence you should expect the solution of patch problems to be more expensive at high order."

"Is there any reason to use the columnwise patches? Also, if the degree is moderate enough and the problem sufficiently benign (say it's just Poisson) then you could try to go entirely matrix free and use P1PC with a Jacobi smoother".

18-04-2023: Draft presentation for CFC2023 posted here as pdf.

15-04-2023: Problems with xvals slicing/matplotlib of Wajiha (one piece I kept); code keeps crashing. Added a flag to comment it out on 16-04-2023. Probably should put the crap in a file.
```Python
mpiexec -n 4 python3 potflow3dper.py
```

11-04-2023: Added latest code (OB). Junho has included max amplification now in another slightly different code. For BLE dtBLEtime=0.5 crashes while dtBLEtime=0.02 still works with energy oscillation. For PFE dtBLEtime=0.02 leads to energy increases while dtBLEtime=0.005 and 0.0025 work. For PFE dtBLEtime=0.01 not known yet. Short time tests.

04-04-2023 OB added energy monitoring in a file with a separate energy-plotting file (printing still not in file); Junho ran SP3 on MPI n=10; first test; latest codes uploaded.

31-03-2023: Dreadful difference remains and results seem to change; I added a flag ntflag; ntflag=0 should be Junho's result (although I have not received a working code confirming that without the minor errors reported below corrected) and ntflag=1 are my results, which follow the scalings written down in draft and WW-paper. I also added a flag ntflag2 to check whether Junho's it can be put for ntflag=1 into a shift of xs but then all hell breaks loose (code flares up and breaks own, no idea why).

30-03-2023: Line 112 fixed by Junho; now it works better but at that=-200 the image is somewhat different from Fig 7a in WW-paper, likely due to an error or difference in O(mu^2) terms or the for-now missing eta corrections in the O(mu^2) terms.

29-03-2023: Fixed a whole set of fd.cosh into np.cosh in SP3 but for that=-200 overflow errors occur since values too large; issue is that Junho's in section 4.3.2. of WW-paper are completely unclear to me; they seem mandatory from a computational viewpoint. Incomprehrensible at the moment. Latest code updated.

29-03-2023: Code updated to date; typo/error in definition of Fx in SP2. ()**(1/3) and not ()*(1/3), although it was in a mu^2 correction term. SP3 issue still remaining but I a, checking and added comments. Plus an error in SP3 fixed; XX1 in one spot should  have been XX2.

28-03-2023: SP3 initial condition seems to work for that0=0 but not that0=-200 (see "JUNHO" in code). Any ideas how to deal with the shifting?

27-03-2023: Works. Junho Choi runs code on HPC. See Appendix and soliton paper. Buys implementing SP3.

25-03: Updated code. Maybe SP2 works. Put on HPC? Put depth dependence in U0(y,z). Tested SP1: converges in time now.

23-03 Updated code. Switch off time-loop for plots. Now segmentation error on SP2.

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


