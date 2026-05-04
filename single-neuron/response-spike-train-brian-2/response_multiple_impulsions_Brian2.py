# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 02:38:31 2026
Note: disabled the User Module Reloader (UMR) in the Python interpreter's 
preferences so as to have better stability and avoid parasite warnings. The
user is encouraged to do the same.
@author: Nicolas Zadeh
"""

# Main packages
import numpy as np

# Brian2 package
from brian2 import (
    NeuronGroup,
    StateMonitor,
    SpikeMonitor,
    start_scope,
    run,
    defaultclock,
    mV,
    volt,
    Mohm,
    farad,
    amp,
    second,
    ms,
    nA,
)

# Plotting packages
import matplotlib.pyplot as plt

from matplotlib import ticker

# Parameters
u_F=23*mV
u_R=10*mV
u_L=0

# From Verechtchaguina2007 
R=500.0*Mohm
C=2.1*10**(-10)*farad
R_L=25.0*Mohm
L=2.5*10**6*volt*second/amp

T=1000

start_scope()

tau=R*L*C/(L+R_L*R*C)
omega_0=np.sqrt((R+R_L)/(R*L*C))
ome0=(R+R_L)/(R*L*C)

a=R_L*ome0
b=1/C

if (1 / tau**2 - 4 * omega_0**2) < 0:
    print("Oscillatory framework")
else:
    raise RuntimeError("Non-oscillatory framework")

eqs = '''   
dv/dt = -v/tau-ome0*x+a*I+b*J: volt/second
dx/dt = v:volt
dI/dt= J :amp
J:amp/second
'''

reset = '''
x= u_R
v= 0*volt/second
'''

defaultclock.dt = 0.1 * ms

G = NeuronGroup(1, eqs, method='euler', threshold='x>u_F', reset=reset)
M = StateMonitor(G, 'x', record=0)
N= StateMonitor(G, 'I', record=0)
spikemon = SpikeMonitor(G)

G.x=0*mV

# Amplitude of the pulses
diff_input_pulse=1.81*10**-6
diff_input_pulse_inhib=2*diff_input_pulse

# Excitatory part

run(50*ms)
G.J = diff_input_pulse*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = -diff_input_pulse*amp/second
run(1*ms)
# run(499*ms)
G.J = 0*amp/second
run(495*ms)
G.J = diff_input_pulse*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = -diff_input_pulse*amp/second
run(1*ms)
# run(499*ms)
G.J = 0*amp/second
run(75*ms)
G.J = diff_input_pulse*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = -diff_input_pulse*amp/second
run(1*ms)
# run(499*ms)
G.J = 0*amp/second
run(495*ms)
G.J = diff_input_pulse*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = -diff_input_pulse*amp/second
run(1*ms)
# run(499*ms)
G.J = 0*amp/second
run(140*ms)
G.J = diff_input_pulse*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = -diff_input_pulse*amp/second
run(1*ms)
# run(499*ms)
G.J = 0*amp/second
run(495*ms)

# Inhibitory part

G.J = -diff_input_pulse_inhib*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = +diff_input_pulse_inhib*amp/second
run(1*ms)
G.J = 0*amp/second
run(140.5*ms)
G.J = -diff_input_pulse_inhib*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = +diff_input_pulse_inhib*amp/second
run(1*ms)
G.J = 0*amp/second
run(140.5*ms)
G.J = -diff_input_pulse_inhib*amp/second
run(1*ms)
G.J = 0*amp/second
run(1*ms)
G.J = +diff_input_pulse_inhib*amp/second
run(1*ms)
G.J = 0*amp/second
run(495*ms, report='text')


# Text fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
})

# Construction of the plot

fig = plt.figure(figsize=(6.2, 4.0))
gs = fig.add_gridspec(2, 1, hspace=0.0)
axs = gs.subplots(sharex=True)

# Membrane potential graph 
axs[0].plot(M.t/ms, M.x[0]/mV, lw=1.2)
axs[0].axhline(y=u_F/mV, ls='--', c='k', lw=0.6)

# Drawn "by hand" spikes, in the spirit of Izhikhevich
for t in spikemon.t:
    axs[0].vlines(x=t/ms, ymin=u_F/mV, ymax=1.19*np.max(M.x[0]/mV),
                  color="black", lw=1)

axs[0].set_ylabel("Membrane potential\n(mV)", fontsize=10)

axs[0].set_yticks([-20, 0, u_R/mV, u_F/mV])

def format_func(value, pos=None):
    if np.isclose(value, -20):
        return r"$-20$"
    elif np.isclose(value, 0):
        return r"$0$"
    elif np.isclose(value, u_R/mV):
        return r"$u_R$"
    elif np.isclose(value, u_F/mV):
        return r"$u_F$"
    return ""

axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(format_func))

axs[0].set_ylim([
    1.2*np.min(M.x[0]/mV),
    1.25*np.max(M.x[0]/mV)
])

# Input current graph
axs[1].plot(M.t/ms, N.I[0]/nA, lw=1.2)

axs[1].set_ylabel("Input current\n(nA)", fontsize=10)
axs[1].set_xlabel("Time (ms)", fontsize=10)

axs[1].set_yticks([-4, -2, 0, 2])
axs[1].set_yticklabels([r"$-4$", r"$-2$", r"$0$", r"$2$"])
axs[1].set_ylim([
    1.2*np.min(N.I[0]/nA),
    1.3*np.max(N.I[0]/nA)
])

# Letters corresponding to the different types of stimuli

y_offset = 0.55   # vertical position in nA, adjust if needed

axs[1].text(50,   -y_offset, "a", ha="center", va="top", fontsize=10)
axs[1].text(595,  -y_offset, "b", ha="center", va="top", fontsize=10)
axs[1].text(1200, -y_offset, "c", ha="center", va="top", fontsize=10)
axs[1].text(1906,y_offset, "d", ha="center", va="bottom", fontsize=10)

# Alignment of the left labels
for ax in axs:
    ax.yaxis.set_label_coords(-0.09, 0.5)
    ax.tick_params(labelsize=10)
    ax.label_outer()
    
    
plt.savefig("BrianR&FSingleNeuron.pdf", format="pdf")
plt.savefig("BrianR&FSingleNeurontight.pdf", format="pdf", bbox_inches="tight")

plt.tight_layout()
plt.show()