# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:03:33 2026
The program should also be able to deal with mutliple spikes, should the
trajectories ask for it (not the case here)
@author: Nicolas Zadeh
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters

# initial condition
x0 = 0.5      # mV
v0 = 2.1      # mV/s

# physical / model parameters
I = 1.6e-1
C = 1.0
R_L = 1.0
omega0 = 2.5  # rad/s
tau = 3.0     # s
T = 12.0   # s

# threshold / reset
u_F = 1.0
u_R = -0.3

# plotting / integration options
dt_dense = 0.002
max_step = 0.01

# style for reset segments
reset_ls = (0, (3, 3))   # dashed; replace by ":" for dotted

# Derived constants

forcing = (R_L * omega0**2 - 1.0 / (C * tau)) * I
x_eq = forcing / omega0**2

print(f"forcing = {forcing:.6f}")
print(f"equilibrium x_eq = {x_eq:.6f}")

# ODE

def rhs(t, y):
    x, v = y
    dxdt = v
    dvdt = -(1.0 / tau) * v - omega0**2 * x + forcing
    return [dxdt, dvdt]

def event_threshold(t, y):
    return y[0] - u_F

event_threshold.terminal = True
event_threshold.direction = 1.0

# Helper: work-around to plot the trajectories the way we want

def hybrid_trajectory_with_reset(x_init, v_init, T):
    """
    Builds the hybrid trajectory with resets.

    Continuous pieces are separated by 'nan's so matplotlib does not
    connect them with artificial straight segments.
    """
    # continuous part for time plot
    t_cont = []
    x_cont = []
    v_cont = []

    # continuous part for phase plot
    phase_x_cont = []
    phase_v_cont = []

    # jump segments stored separately
    time_jump_segments = []    # list of ([t_hit, t_hit], [u_F, u_R])
    phase_jump_segments = []   # list of ([x_hit, u_R], [v_hit, 0.0])

    spike_times = []

    t0 = 0.0
    y0 = np.array([x_init, v_init], dtype=float)
    first_piece = True

    while t0 < T:
        sol = solve_ivp(
            rhs,
            [t0, T],
            y0,
            events=event_threshold,
            dense_output=True,
            max_step=max_step,
            rtol=1e-9,
            atol=1e-11,
        )

        t_end = sol.t[-1]
        t_piece = np.arange(t0, t_end, dt_dense)
        if len(t_piece) == 0 or t_piece[-1] < t_end:
            t_piece = np.append(t_piece, t_end)

        y_piece = sol.sol(t_piece)
        x_piece = y_piece[0]
        v_piece = y_piece[1]

        # separate consecutive continuous arcs
        if not first_piece:
            t_cont.append(np.nan)
            x_cont.append(np.nan)
            v_cont.append(np.nan)

            phase_x_cont.append(np.nan)
            phase_v_cont.append(np.nan)

        t_cont.extend(t_piece.tolist())
        x_cont.extend(x_piece.tolist())
        v_cont.extend(v_piece.tolist())

        phase_x_cont.extend(x_piece.tolist())
        phase_v_cont.extend(v_piece.tolist())

        first_piece = False

        # no threshold crossing
        if len(sol.t_events[0]) == 0:
            break

        # threshold hit
        t_hit = sol.t_events[0][0]
        spike_times.append(t_hit)

        y_hit = sol.sol(t_hit)
        x_hit = float(y_hit[0])
        v_hit = float(y_hit[1])

        # dashed reset segments
        time_jump_segments.append(([t_hit, t_hit], [u_F, u_R]))
        phase_jump_segments.append(([x_hit, u_R], [v_hit, 0.0]))

        # restart from reset
        t0 = t_hit + 1e-10
        y0 = np.array([u_R, 0.0], dtype=float)

    return (
        np.array(t_cont),
        np.array(x_cont),
        np.array(v_cont),
        np.array(phase_x_cont),
        np.array(phase_v_cont),
        time_jump_segments,
        phase_jump_segments,
        spike_times,
    )

# Build data

(
    t_hybrid,
    x_hybrid,
    v_hybrid,
    phase_x_hybrid,
    phase_v_hybrid,
    time_jump_segments,
    phase_jump_segments,
    spike_times,
) = hybrid_trajectory_with_reset(x0, v0, T)

print("Spike times:", [round(s, 4) for s in spike_times])

# Plot settings

plt.rcParams.update({
    "font.size": 12,
    "mathtext.fontset": "cm",
    "font.family": "serif"
})

blue = "#5b7dbd"
red = "#d94841"
purple = "#7b2cbf"

# Figure 1: phase portrait

fig1, ax = plt.subplots(figsize=(4.2, 3.7))

# continuous pieces only
ax.plot(phase_x_hybrid, phase_v_hybrid, color=blue, lw=1.0)

# dashed reset jumps
for xs, vs in phase_jump_segments:
    ax.plot(xs, vs, color=blue, lw=1.0, ls=reset_ls)

# threshold
ax.axvline(u_F, color=red, lw=1.1)
ax.text(u_F + 0.02, 2.0, r"$x=u_F$", color=red, fontsize=13)

# reset point
ax.plot([u_R], [0.0], "o", color=purple, ms=4)
ax.text(u_R - 0.08, -0.28, r"$u_R$", color=purple, fontsize=14)

# initial point label
ax.text(x0 + 0.02, v0 + 0.12, r"$(x_0,v_0)$", color=blue, fontsize=13)

# axes through origin
ax.axhline(0, color="0.7", lw=0.8)
ax.axvline(0, color="0.7", lw=0.8)

xmin, xmax = -0.35, 1.15
ymin, ymax = -2.3, 2.45
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)

ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ax.annotate(
    "",
    xy=(xmax, 0),
    xytext=(xmax - 0.001, 0),
    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
)
ax.annotate(
    "",
    xy=(0, ymax),
    xytext=(0, ymax - 0.001),
    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
)

ax.text(xmax - 0.05, -0.3, r"$x$", fontsize=14)
ax.text(0.03, ymax - 0.10, r"$v$", fontsize=14)

fig1.tight_layout()
fig1.savefig(
    "Single neuron ODE rebound negative reset.pdf",
    format="pdf",
    bbox_inches="tight"
)

# Figure 2: x(t)

fig2, ax = plt.subplots(figsize=(4.2, 3.7))

# continuous pieces only
ax.plot(t_hybrid, x_hybrid, color=blue, lw=1.0)

# dashed vertical reset jumps
for ts, xs in time_jump_segments:
    ax.plot(ts, xs, color=blue, lw=1.0, ls=reset_ls)

# threshold
ax.axhline(u_F, color=red, lw=1.1)
ax.text(T - 2.0, u_F + 0.06, r"$x=u_F$", color=red, fontsize=13)

# labels
ax.text(-0.92, x0, r"$x_0$", color=blue, fontsize=13)
ax.text(-0.98, u_R, r"$u_R$", color=purple, fontsize=13)

# axes through origin
ax.axhline(0, color="0.7", lw=0.8)
ax.axvline(0, color="0.7", lw=0.8)

ax.set_xlim(0, T)
ax.set_ylim(min(-0.9, u_R - 0.15), 1.2)

ax.spines[:].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ax.annotate(
    "",
    xy=(T, 0),
    xytext=(T - 0.001, 0),
    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
)
ax.annotate(
    "",
    xy=(0, 1.2),
    xytext=(0, 1.199),
    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
)

ax.text(T - 0.22, -0.16, r"$t$", fontsize=14)
ax.text(-1.4, 1.13, r"$x(t)$", fontsize=14)

# ax.text(-0.16, 1.0, r"$1$", fontsize=11)

fig2.tight_layout()
fig2.savefig(
    "Single neuron ODE marginal rebound negative reset.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()