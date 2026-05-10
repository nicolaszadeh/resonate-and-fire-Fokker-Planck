Each of the scripts bearing a solver name produces a .csv spreadsheet as an output containing the runtime of said scheme, for multiple values of (Nt,Nx,Nv).
The user wishing to modify (Nt,Nx,Nv) can go in the code and change 'seq_Nt' as well as 'seq_n_xv', the sequence of grid depths.

A boolean 'USE_SOUND' is at the disposal of the user (by default at 'True'), which allows to play success or failure melodies, which happens to be especially useful in the case of very long computations for example for large 'n' and element of 'seq_Nt'. 

A tree structure adapted to the outputs is automatically created at launch of any of these scripts.
