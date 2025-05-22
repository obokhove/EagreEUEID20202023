Coupled beach and wave: basic codes and analysis codes

Main code(s) FG&YL: python3 coupled_tank.py (creates data files).
See also theses Floriane Gidel (2018) and Yang Lu (2025, pending) UNiversity of Leeds.

Run time OB macbook 2020 via Dockers installation FD: Before time loop: 102.00s Computation time: 2h15min21.46s Total:8121.46s Memory usage less than ~406MB

Run time YL macbook Intel installation FD: ~1:27hr Memory usage less than ~300MB

Analyse energy and waterline (OB: NB. current waterline analysis is incorrect 22-05-2025) run in main directory: python3 pp_energy.py 

Python and paraview files to be run (I guess in data directory): dwswbeach2025.pvsm, dwswbeachwave2025.pvsm, dwswbeachwave2025.py (OB: untested)

Paraview instructions in Paraview5.docx file (OB; untested)
