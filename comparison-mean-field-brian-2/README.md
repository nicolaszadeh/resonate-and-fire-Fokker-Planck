The presented scripts give as an output 3 comparison graphs, for the activity of the network, the mean voltage, as well as the final voltage-density at the final time in the experiment, where both the numerical solution to the pde obtained via our scheme appear and the result of large scale Brian 2 run.

In the script ending with "std" the initialization is made after a continuous gaussian, and when changing the number of runs, the mean of the observables is plotted, as well as one standard deviation around it.

In the script without the "std" the initialization is made after the initial pdf, on the grid, and it is made for a single run.

Of course Brian 2 will need to be implemented on the user's computer. More information on Brian 2 can be found in the README file at the root of the repository.

The user can either use the 'environment.yml' file which will take care of Brian 2 automatically, or, if they want to install the packages one by one, go for a manual install, through
```bash
pip install brian2
```
or
```bash
conda install -c conda-forge brian2
```

The parameters by default are quite demanding (the goal was to show adequation between the kinetic theory and the large scale simulation, so there needed to be fine precision for the pde solver and important number of neurons for the point network at the same time). We recommend the user which would just like to get familiar with the program and its outputs to comment these parameters out and use those above in the script (this concerns the parameters n, Nt_user, NE).
