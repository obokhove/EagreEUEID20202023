 ## Coupled beach and wave: basic codes and analysis codes

Main code(s) FG&YL: python3 coupled_tank.py (creates data files).
See also theses Floriane Gidel (2018) and Yang Lu (2025) University of Leeds.

Run times:
- OB macbook 2020 via Dockers installation FD: Before time loop: 102.00s Computation time: 2h15min21.46s Total:8121.46s Memory usage less than ~406MB . 
- YL macbook Intel installation FD: ~1:27hr Memory usage less than ~300MB
- M2 macbook Docker: 59:38min; memory usage 459.84MB.
- M2 macbook Native: 39:23min; memory usage: 387.98MB. So 1.5x faster on M2 macbook.

Analyse energy and waterline (OB: NB. h=0 waterline analysis updated 22-05-2025) run in main directory:
- Figs. 4 and 7 (referring to pdf document [29-07-2025]): python3 pp_energy.py (check output in main code)
- Fig. 8: python3 pp_energy1234.py (check multiple h=c outputs in main code)

Figs. 5 and 6:
- Python and paraview files to be run (e.g., in data directory): dwswbeach2025.pvsm, dwswbeachwave2025.pvsm, dwswbeachwave2025.py (OB)

Paraview instructions in Paraview5.docx file (OB)

Added run-up heights to the code in order to assess where the "water line" resides as it does not reside at the h=0-contour when water is receding. Added new energy-plotting and main files. For comparison of PF and MSA, using h=0 may be okay, but it is not the waterline; percentage change in water line xw is artificial since xw average value is artificial. See !["Water line" proxy's](energy_and_xw.png)
