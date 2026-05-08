<h1 align="center">
Kinetic Fokker-Planck equation for a network of noisy resonate-and-fire neurons
</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-blue">
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

<div align="left">
  <img align="right" src="/additional-figures-and-videos/figures/reset_driven_snapshots.png" width="45%">

Code, relevant figures and videos used to illustrate and support:

- the derivation of a PDE model for a network of noisy resonate-and-fire neurons
- the original associated numerical mass- and positivity-preserving solver
- several related observations and conjectures
<br><br><br><br><br>
</div>

# Description of the model
The model arises from a heuristic mean-field limit of [Izhikhevich's resonate and fire model](https://doi.org/10.1016/S0893-6080(01)00078-8) in a small-kick-size and large number of neurons approximation.
We denote $u_{\rm F}$ the firing voltage threshold and $u_{\rm R}$ the reset voltage.

It takes the form of a kinetic Fokker-Planck equation, with $x\leq u_{\rm F}$ being the voltage variable and $v$ the associated velocity, the unknown being a probability density function f(x,v,t). 
The equation reads

```math
\partial_t f + v \partial_x f 
+ \left(-\frac{v}{\tau} - \omega_0^2 x + b\bigl(N(t)+\nu_{\rm ext}\bigr)\right)\partial_v f
- \frac{f}{\tau} 
- a(t)\partial_v^2 f
= N(t)\,\delta_{(u_R,0)},
```

with $N(t)$ being the activity of the network, as in the average number of spikes per neuron per second, and $a(t)=a_0 + a_1 N(t)$.<br>

The expression of $N$ is:
```math
N(t):=\int_{v>0}vf(u_F,v,t)\,{\rm d}v.
```
# Numerical study
The numerical study is done in a domain 
```math
E:=(x_{\rm min},x_{\rm max})\times (-V,V).
```
with $x_{\rm max}=u_{\rm F}$.

<br>

We implement inflow homogeneous Dirichlet boundary conditions on the side edges:
```math
f(u_{\rm F},v<0,t)=0,
```
```math
f(x_{\rm min},v>0,t)=0,
```

<br>


as well as non-influx Robin boundary conditions, which, with, $V$ large enough is, for all $x$ in $(x_{\rm min},x_{\rm max})$:
```math
f(x,V,t)-a(t)\partial_v f(x,V,t)=0,
```
```math
f(x,-V,t)-a(t)\partial_v f(x,-V,t)=0.
```
<br>

The Dirac delta is approximated by a very concentrated Maxwellian of mass $1$ centered around point $(u_{\rm R},0)$.

<br>

The used method is an original upwind semi-implicit finite differences method with an uniform mesh in $x,v,t$. Mass and positivity preservation properties under mild dynamic conditions can be rigorously proven.

# Contents of the repository

As of May the 8th, 2026, the two-level-tree structure of the repository (barring the README files) is the following:
```text
├── additional-figures-and-videos
│   ├── figures
│   └── videos
├── comparison-mean-field-brian-2
│   └── comparison_Brian_mean_field_N_V_final_density.py
├── Fokker-Planck-equation
│   ├── experimental-order-of-convergence
│   ├── exploration-of-different-regimes
│   └── vector-field
└── single-neuron
    ├── response-spike-train-brian-2
    └── solution-ODE
```

# Broad contents of the sub-repositories
- _additional-figures-and-videos_ : a few especially representative illustrative contents
- _comparison-mean-field-brian-2_ : a comparison of the pde model/solver and the Brian 2 solver [Elife paper](https://elifesciences.org/articles/47314), [Brian2 documentation](https://brian2.readthedocs.io/en/stable/index.html)
- _experimental-order-of-convergence_ : scripts allowing to compare the performance of the semi-implicit upwind scheme with other classic methods
- _exploration-of-different-regimes_ : scripts giving access to information regarding the evolution of solutions in different regimes, as well as long-time behaviour
- _vector-field_ : a script plotting the drift vector field in the Fokker-Planck equation
- _response-spike-train-brian-2_ : code used to illustrate the behaviour of neurons thanks to the Brian 2 solver
- _solution-ODE_ : solver giving the trajectory in phase space of a single neuron following the deterministic ODE from Izhikhevich's model

Functionals of interest, such as mean voltage, activity, entropy, Fisher information (these last two quantities also being presented in a "relative to the long-time behaviour" form) are also displayed. All these outputs can be produced or not, depending on user-chosen booleans (by default at true). The long-time behaviour can be displayed in 2d or 3d form.

# How to use
- The user just has to download the sub-repository containing the code they are interested in.
- The execution of the Python code will produce results on their computer.
- The parameters are by default quite demanding, they can of course edit them to obtain less precision, but also different results from ours if they want to get different initial conditions or model parameters for example.
- May they want to launch the scripts with demanding parameters, we implemented sound subroutines inside them which announce when the scripts are over (with a different melody depending on whether the execution was successful or not).
- We insist on the fact that the codebase was intentionally organized so that each sub-repository can be used independently, without requiring the others.

# Requirements
An environment.yml file is available at the root of the repository. The command 'conda env create -f environment.yml' (without the quotes) allows to reproduce the exact environment used to obtain our results. Without this, opening it in a text editor already allows to consult the list of packages used.

# Compatibility
All the codebase has been extensively tested on Windows, Linux and MacOS. 

# Citation

If you use this repository in academic work, please cite the associated article when available.

In the meantime, please reference the repository itself:

```text
Nicolas Zadeh,
"Kinetic Fokker–Planck equation for a network of noisy resonate-and-fire neurons",
github.com/nicolaszadeh/resonate-and-fire-Fokker-Planck/,
2026.
```
