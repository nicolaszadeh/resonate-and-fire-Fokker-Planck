Each of the scripts bearing a scheme name produces a .csv spreadsheet as an output containing information pertaining to the experimental order of convergence of said scheme as well as its runtime.
The parameters by default are demanding, I would advise the reader to first run them by commenting out the grid's depth 'n' and the sequence of successive time-points 'seq_Nt' to replace them with the more reasonable values offered in commentary right above them. 

A boolean 'USE_SOUND' is at the disposal of the user (by default at 'True'), which allows to play success or failure melodies, which happens to be especially useful in the case of very long computations for example for large 'n' and element of 'seq_Nt'. 

The parallel launcher is particular: it runs by default all 4 schemes _in parallel_, which allows to use the multi-core technology of modern processors which we couldn't use to its full extent most of the time with our solver. The scripts all need to be in a same folder.
The output of the parallel launcher is log files (especially useful in case of non-completion to understand the source of the error), as well as .csv spreadsheets. 

A tree structure adapted to the outputs is automatically created at launch of any of these scripts.
