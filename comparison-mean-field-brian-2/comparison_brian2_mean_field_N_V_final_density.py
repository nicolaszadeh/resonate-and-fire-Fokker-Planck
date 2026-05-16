# -*- coding: utf-8 -*-
"""
April the 19th, 2026
comparison code without rescaling:
- PDE model directly in physical units
- Brian2 population model in physical units
- outputs 3 comparison graphs:
    1) activity N(t)
    2) mean voltage x(t)
    3) final x-density at T

Units used here:
- t in seconds
- x in volt
- v in volt/s
- N in Hz

Brian uses the microscopic interpretation:
each spike updates both I and v.
Note: a misleading error signal usually appears in our IDE,
it doesn't prevent the program from running well.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.linalg import solve_banded
from datetime import datetime
import os
import time
import math
import warnings
import traceback
import sys

warnings.filterwarnings("ignore", category=UserWarning)

# User choices

RUN_BRIAN = True

SAVE_PDF = True
SAVE_PNG = False
DIAG_PLOT_DPI = 300

# Time horizon (seconds)
T_phys = 1
T = T_phys

# Grid level for PDE
# n=4
# n=12
n = 30

# PDE time steps
# Nt_user = 5001
# Nt_user = 10001
Nt_user = 100001

# Brian parameters
N_RUNS=1
Nt_Brian = Nt_user
BRIAN_SAMPLE_DT = T_phys / Nt_Brian   # seconds

# NE = 8000
# NI=2000
NE = 80000
NI = 20000
N_BRIAN = NE + NI

# Brian activity bin width
BRIAN_RATE_BIN_MS = 5.0
BIN_WIDTH = BRIAN_RATE_BIN_MS * 1e-3   # seconds

# ============================================================
# Parameters
# ============================================================

# 1 volt in volts
U0_volt = 1

# Reset
# u_F = 15.0 * U0_volt
u_F=9

# Domain
# IMPORTANT: here v_min, v_max are interpreted in volt/s

x_min = -9
x_max = u_F

V = 25

v_min = -V * U0_volt
v_max = V * U0_volt

size_x = (x_max - x_min)
size_v = (v_max - v_min)

# Initial condition
x10=4
v10=15

sigma_init_x = 3 * U0_volt
sigma_init_v = 3 * U0_volt

# Source
u_R=4
sigma_rho_x = 0.001 * U0_volt
sigma_rho_v = 0.001 * U0_volt

X = max(u_F, abs(x_min))
V = max(abs(v_min), abs(v_max))

# Physical parameters

R_phys = 0.8
R_L_phys = 0.8
C_phys = 1
L_phys = 2
tau_syn_phys = 10

# External rate
nu_ext_phys = 0.1

# Connectivity
C_E_ext = 800
C_I_ext = 200
C_E_int = 800
C_I_int = 200

# Synaptic areas, J is small and it's critical
J = 0.0001
J_E_ext = J
J_I_ext = J
J_E_int = J
J_I_int = J

# Derived oscillator parameters

omega0_phys = math.sqrt((R_phys + R_L_phys) / (R_phys * L_phys * C_phys))
tau_phys = (R_phys * L_phys * C_phys) / (L_phys + R_L_phys * R_phys * C_phys)
beta_phys = omega0_phys * tau_phys

# Directly in physical units
omega_0 = omega0_phys
tau = tau_phys

# Corrected coefficients

b_ext_phys = (C_E_ext * J_E_ext - C_I_ext * J_I_ext) / (C_phys * tau_syn_phys)
b_int_phys = (C_E_int * J_E_int - C_I_int * J_I_int) / (C_phys * tau_syn_phys)

a0_phys = (
    (C_E_ext * J_E_ext**2 + C_I_ext * J_I_ext**2)
    / (2.0 * C_phys**2 * tau_syn_phys**2)
) * nu_ext_phys

a1_phys = (
    (C_E_int * J_E_int**2 + C_I_int * J_I_int**2)
    / (2.0 * C_phys**2 * tau_syn_phys**2)
)

nu = nu_ext = nu_ext_phys
b_ext = b_ext_phys
b_int = b_int_phys
b = b_ext
a_0 = a0_phys
a_1 = a1_phys

# Utilities

DTYPE = np.float64

def pretty_time(s):
    m, s = divmod(float(s), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}"

def style_axes(ax, yfmt="%.2f"):
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(yfmt))
    ax.tick_params(axis='both', which='both', direction='out', length=2, width=0.8)

def fmt_float_for_filename(x):
    return f"{x:.2e}".replace('.', 'p').replace('-', 'm').replace('+', '')

# Grid

Xmax = max(abs(x_min), abs(x_max))
Vmax = max(abs(v_min), abs(v_max))

Nt = Nt_user
delta_t = np.float64(T / (Nt - 1))

print("PHYSICAL PARAMETERS (NO RESCALING)")
print(f"omega0_phys = {omega0_phys:.6g} s^-1")
print(f"tau_phys    = {tau_phys:.6g} s")
print(f"beta_phys   = {beta_phys:.6g}")
print()
print(f"b_ext_phys(SI) = {b_ext_phys:.6e} V/s^2")
print(f"b_int_phys(SI) = {b_int_phys:.6e} V/s^2")
print(f"a0_phys(SI)    = {a0_phys:.6e} V^2/s^3")
print(f"a1_phys(SI)    = {a1_phys:.6e} V^2/s^3")
print()
print(f"nu_ext      = {nu_ext:.6e} Hz")
print()
print(f"T = {T:.6e} s")
print(f"Nt used = {Nt}, dt = {delta_t:.6e} s")

if (1.0 / tau**2 - 4.0 * omega_0**2) < 0.0:
    print("Oscillatory regime (underdamped).")
else:
    print("Warning: not in oscillatory regime.")

# Snapshot export parameters

SAVE_SNAPSHOTS = True
NUM_SNAPSHOTS = 9
SAVE_SNAPSHOT_PNG = False
SNAPSHOT_DPI = 400

# Extra diagnostics plots

SAVE_ACTIVITY_PLOT = True
SAVE_ENTROPY_PLOT = False
SAVE_EXPECTATION_PLOT = True
DIAG_PLOT_DPI = 400

# Basic PDE utilities

def mass_of(f):
    return delta_x * delta_v * np.sum(f, dtype=np.float64)

def entropy_of(f):
    f_pos = f[f > 0.0]
    return np.float64(
        -delta_x * delta_v * np.sum(f_pos * np.log(f_pos), dtype=np.float64)
    )

def mean_x_of(f):
    return np.float64(
        delta_x * delta_v * np.sum(x_col * f, dtype=np.float64)
    )

def build_snapshot_indices(num_frames, num_snapshots):
    if num_snapshots < 2:
        return np.array([0], dtype=int)
    return np.linspace(0, num_frames - 1, num_snapshots, dtype=int)

np.set_printoptions(precision=25)

# Globals

Nx = Nv = None
delta_x = delta_v = None
dt_over_dx = dt_over_dv = None

x = v = None
x_col = v_row = None

f_initial = rho = None
i_F = j_0 = None

j_pos_x = j_neg_x = None
interior_rows = None

ab_row = None
J_full = None
x_interior_col = None
v_interior_row = None

# Init

def init_xv(n):
    global Nx, Nv, delta_x, delta_v, dt_over_dx, dt_over_dv
    global x, v, x_col, v_row
    global f_initial, rho, i_F, j_0
    global j_pos_x, j_neg_x, interior_rows
    global ab_row, J_full, x_interior_col, v_interior_row

    v_factor = 1

    number_points_space = int((n + 1) * size_x / U0_volt + 1)
    number_points_velocity = int((n + 1) * (size_v / U0_volt) / v_factor + 1)

    Nx = number_points_space - 2
    Nv = number_points_velocity - 2

    delta_x = np.float64(size_x / (Nx + 1))
    delta_v = np.float64(size_v / (Nv + 1))

    print(f"Nx = {Nx}, Nv = {Nv}, dx = {delta_x}, dv = {delta_v}")

    dt_over_dx = delta_t / delta_x
    dt_over_dv = delta_t / delta_v

    x = np.linspace(x_min, x_max, Nx + 2, dtype=np.float64)
    v = np.linspace(v_min, v_max, Nv + 2, dtype=np.float64)

    def index_x(point):
        return int((point - x_min) / delta_x)

    def index_v(point):
        return int((point - v_min) / delta_v)

    i_F = index_x(u_F)
    j_0 = index_v(0)

    x_col = x[:, None]
    v_row = v[None, :]

    interior_rows = slice(1, i_F)
    x_interior_col = x[1:i_F, None]
    v_interior_row = v[None, :]

    j_pos_x = slice(j_0 + 1, Nv + 2)   # v > 0
    j_neg_x = slice(0, j_0)            # v < 0

    # initial condition

    inv2s2_init_x = 1.0 / (2.0 * sigma_init_x**2)
    gx_init = np.exp(-(x - x10)**2 * inv2s2_init_x)

    inv2s2_init_v = 1.0 / (2.0 * sigma_init_v**2)
    gv_init = np.exp(-(v - v10)**2 * inv2s2_init_v)
    f0 = np.multiply.outer(gx_init, gv_init)

    f0[0, j_0:] = 0.0
    f0[i_F, :j_0 + 1] = 0.0

    f_initial = f0 / (delta_x * delta_v * np.sum(f0, dtype=np.float64))
    f_initial *= 1.0 / (delta_x * delta_v * np.sum(f_initial, dtype=np.float64))

    print("Initial mass =", mass_of(f_initial))

    # source

    inv2s2_src_x = 1.0 / (2.0 * sigma_rho_x**2)
    inv2s2_src_v = 1.0 / (2.0 * sigma_rho_v**2)
    gx_src = np.exp(-(x - u_R)**2 * inv2s2_src_x)
    gv_src = np.exp(-(v - 0.0)**2 * inv2s2_src_v)
    rho0 = np.multiply.outer(gx_src, gv_src)

    rho0[0, j_0:] = 0.0
    rho0[i_F, :j_0 + 1] = 0.0

    rho = rho0 / (delta_x * delta_v * np.sum(rho0, dtype=np.float64))

    # row-wise implicit matrix A
    ab_row = np.zeros((3, Nv + 2), dtype=np.float64)

    # j-array reused in critical-strip construction
    J_full = np.arange(Nv + 2, dtype=np.int64)[None, :]

# Implicit row matrix in v

def build_ab_row(N):
    alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

    ab_row.fill(0.0)

    ab_row[1, :] = 1.0 + 2.0 * alpha
    ab_row[0, 1:] = -alpha
    ab_row[2, :-1] = -alpha

    ab_row[1, 0] = 1.0 + alpha
    ab_row[1, -1] = 1.0 + alpha

    return ab_row

# Transport operator B

def apply_B_2d(f, N):
    out = f.copy()

    # Horizontal transport in x

    if j_0 + 1 < Nv + 2:
        coeff_pos = (-v[j_pos_x] * dt_over_dx)[None, :]
        out[1:i_F + 1, j_pos_x] += coeff_pos * (
            f[1:i_F + 1, j_pos_x] - f[0:i_F, j_pos_x]
        )

    if j_0 > 0:
        coeff_neg = (-v[j_neg_x] * dt_over_dx)[None, :]
        out[0:i_F, j_neg_x] += coeff_neg * (
            f[1:i_F + 1, j_neg_x] - f[0:i_F, j_neg_x]
        )

    # Vertical transport in v on interior x-rows
    if i_F > 1:
        f_int = f[1:i_F, :]
        out_int = out[1:i_F, :]

        muv = - (omega_0**2) * x_interior_col - v_interior_row / tau + b * (nu + N)
        beta_v = -muv * dt_over_dv

        out_int += (delta_t / tau) * f_int

        jc = (((tau * (-omega_0**2 * x[1:i_F] + b * (nu + N)) + V) / delta_v)).astype(np.int64)
        crit_mask = (J_full == jc[:, None]) | (J_full == (jc[:, None] + 1))

        out_int += np.where(
            crit_mask & (muv > 0.0),
            -(beta_v + delta_t / tau) * f_int,
            0.0
        )

        out_int += np.where(
            crit_mask & (muv < 0.0),
            (beta_v - delta_t / tau) * f_int,
            0.0
        )

        if Nv >= 1:
            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] > 0.0,
                beta_v[:, 1:Nv + 1] * (f_int[:, 1:Nv + 1] - f_int[:, 0:Nv]),
                0.0
            )

            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] < 0.0,
                beta_v[:, 1:Nv + 1] * (f_int[:, 2:Nv + 2] - f_int[:, 1:Nv + 1]),
                0.0
            )

        mu_bottom = - (omega_0**2) * x[1:i_F] - v[0] / tau + b * (nu + N)
        mu_top = - (omega_0**2) * x[1:i_F] - v[Nv + 1] / tau + b * (nu + N)

        out[1:i_F, 0] += -mu_bottom * dt_over_dv * f[1:i_F, 0]
        out[1:i_F, Nv + 1] += mu_top * dt_over_dv * f[1:i_F, Nv + 1]

    return out

# Implicit solve in v

def solve_A_rowwise(D2, N):
    fnew = np.empty_like(D2)

    fnew[0, :] = D2[0, :]
    fnew[i_F, :] = D2[i_F, :]

    if i_F > 1:
        ab = build_ab_row(N)
        rhs = D2[1:i_F, :].T
        sol = solve_banded((1, 1), ab, rhs)
        fnew[1:i_F, :] = sol.T

    return fnew


def compute_activity(f):
    return np.float64(
        delta_v * np.sum(f[i_F, j_0 + 1:Nv + 2] * v[j_0 + 1:Nv + 2], dtype=np.float64)
        - delta_v * np.sum(f[0, 0:j_0] * v[0:j_0], dtype=np.float64)
    )

def step(f):
    N = compute_activity(f)
    D2 = apply_B_2d(f, N) + N * delta_t * rho
    fnew = solve_A_rowwise(D2, N)
    return fnew, N

# Plotting helpers

def save_snapshot(frame2d, time_value, idx, outdir, stem, vmin, vmax):
    fig, ax = plt.subplots(figsize=(6, 4))

    mesh = ax.imshow(
        frame2d.T,
        origin='lower',
        aspect='auto',
        extent=[x[0]/U0_volt, x[-1]/U0_volt, v[0]/U0_volt, v[-1]/U0_volt],
        cmap=colormap,
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest'
    )

    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(r"$f$", rotation=0, labelpad=15)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.label.set_verticalalignment('center')

    ticks = np.linspace(vmin, vmax, 7)
    cbar.set_ticks(ticks)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$", rotation=0, labelpad=15)
    ax.set_title(rf"$t = {time_value:.4f}$")
    ax.set_ylim(-500, 500)

    plt.tight_layout()

    base = os.path.join(outdir, f"{stem}_snapshot_{idx+1:02d}_t{time_value:.4f}")

    pdf_name = base + ".pdf"
    fig.savefig(pdf_name, bbox_inches="tight")

    if SAVE_SNAPSHOT_PNG:
        png_name = base + ".png"
        fig.savefig(png_name, dpi=SNAPSHOT_DPI, bbox_inches="tight")

    plt.close(fig)

# Main PDE simulation

start_time = time.time()

init_xv(n)
print("Start:")
print(f"Nx = {Nx}, Nv = {Nv}, dx = {delta_x}, dv = {delta_v}, dt = {delta_t}")

f = f_initial.copy()

times = np.empty(Nt, dtype=np.float64)
if SAVE_ACTIVITY_PLOT:
    activities = np.empty(Nt, dtype=np.float64)
masses = np.empty(Nt, dtype=np.float64)
if SAVE_ENTROPY_PLOT:
    entropies = np.empty(Nt, dtype=np.float64)
mean_x_values = np.empty(Nt, dtype=np.float64)

if SAVE_SNAPSHOTS:
    snapshot_indices = build_snapshot_indices(Nt, NUM_SNAPSHOTS)
    snapshot_index_set = set(snapshot_indices.tolist())
    snapshot_frames = {}
else:
    snapshot_indices = np.array([], dtype=int)
    snapshot_index_set = set()
    snapshot_frames = {}

times[0] = 0.0
if SAVE_ACTIVITY_PLOT:
    activities[0] = compute_activity(f)
masses[0] = mass_of(f)
if SAVE_ENTROPY_PLOT:
    entropies[0] = entropy_of(f)

mean_x_values[0] = mean_x_of(f)

if 0 in snapshot_index_set:
    snapshot_frames[0] = f.copy()

for k in range(1, Nt):
    f, N = step(f)

    current_time = k * delta_t
    current_entropy = entropy_of(f)
    current_mean_x = mean_x_of(f)

    times[k] = current_time
    if SAVE_ACTIVITY_PLOT:
        activities[k] = N
    if SAVE_ENTROPY_PLOT:
        entropies[k] = current_entropy
    mean_x_values[k] = current_mean_x

    if k in snapshot_index_set:
        snapshot_frames[k] = f.copy()

    current_mass = mass_of(f)

    if abs(current_mass - 1) > 1e-1:
        print(f'At moment k={k},')
        sys.exit("mass conservation failed")

    if (k % 1000 == 0) or (k == Nt - 1):
        print(
            f"k={k}/{Nt-1}, t={current_time:.4f}, "
            f"N={N:.6e}, mass={current_mass:.16f}, mean={current_mean_x/U0_volt:.16f}"
        )

print(f"Simulation done in {time.time() - start_time:.2f} s")

# Run Brian2 

mean_activity_brian = None
tb_bin = None

mean_x_brian_interp = None
brian_times_mean = None

rho_x_brian_final = None
x_hist_centers = None

if RUN_BRIAN:
    print("\n Running Brian2...")

    try:
        from brian2 import (
            prefs, start_scope, defaultclock,
            second, volt, amp, ohm, farad, Hz, hertz,
            NeuronGroup, Synapses, SpikeMonitor,
            PoissonGroup, Network, network_operation
        )

        if N_BRIAN <= 0:
            raise ValueError("N_BRIAN must be > 0 to run Brian2.")
        
        
        x_pde=x.copy()
        v_pde=v.copy()
        prefs.codegen.target = "numpy"

        # fixed objects for binning
        edges = np.arange(0.0, T_phys + BIN_WIDTH, BIN_WIDTH)
        if edges[-1] < T_phys:
            edges = np.append(edges, T_phys)
        # tb_bin = 0.5 * (edges[:-1] + edges[1:])
        # left edges instead of center
        tb_bin = edges[:-1]


        # initial discrete law from PDE initial datum
        prob = f_initial.ravel().astype(float)
        prob /= prob.sum()

        # Brian physical constants
        tau_b = tau_phys * second
        omega0_b = omega0_phys / second
        Cb = C_phys * farad
        RLb = R_L_phys * ohm
        taus = tau_syn_phys * second

        J_E_ext_eff = (J_E_ext / tau_syn_phys) * amp
        J_I_ext_eff = (J_I_ext / tau_syn_phys) * amp
        J_E_int_eff = (J_E_int / tau_syn_phys) * amp
        J_I_int_eff = (J_I_int / tau_syn_phys) * amp

        eqs = '''
        dx/dt = v : volt
        dv/dt = -v/tau_b - (omega0_b**2) * x + (RLb*(omega0_b**2) - 1.0/(Cb*taus)) * I : volt/second
        dI/dt = -I/taus : amp
        '''

        activity_runs = np.empty((N_RUNS, len(tb_bin)), dtype=float)
        mean_x_runs = np.empty((N_RUNS, Nt_Brian), dtype=float)
        rho_x_runs = np.empty((N_RUNS, len(x)), dtype=float)
        brian_times_ref = None

        t_brian0 = time.time()

        for run_id in range(N_RUNS):
            start_scope()
            defaultclock.dt = delta_t * second
            # defaultclock.dt = delta_t / 10 * second

            rng = np.random.default_rng(12345 + run_id)

            G = NeuronGroup(
                N_BRIAN,
                eqs,
                method='euler',
                threshold='x > uF_thr',
                reset='x = uR_reset; v = 0*volt/second',
                namespace={
                    'tau_b': tau_b,
                    'omega0_b': omega0_b,
                    'RLb': RLb,
                    'Cb': Cb,
                    'taus': taus,
                    'uF_thr': u_F * volt,
                    'uR_reset': u_R * volt,
                }
            )

            flat_idx = rng.choice(prob.size, size=N_BRIAN, p=prob)
            ix, jv = np.unravel_index(flat_idx, f_initial.shape)

            G.x = x_pde[ix] * volt
            G.v = v_pde[jv] * volt / second
            G.I = 0.0 * amp

            brian_objects = [G]

            Ge = None
            Gi = None

            if NE > 0:
                Ge = G[:NE]

            if NI > 0:
                Gi = G[NE:NE+NI]

            if NE > 0:
                p_E = min(1.0, C_E_int / NE)
                SE = Synapses(
                    Ge, G,
                    on_pre='v_post += J_E_int_eff/Cb',
                    namespace={
                        'J_E_int_eff': J_E_int_eff,
                        'Cb': Cb
                    }
                )
                SE.connect(p=p_E)
                brian_objects.append(SE)

            if NI > 0:
                p_I = min(1.0, C_I_int / NI)
                SI = Synapses(
                    Gi, G,
                    on_pre='v_post -= J_I_int_eff/Cb',
                    namespace={
                        'J_I_int_eff': J_I_int_eff,
                        'Cb': Cb
                    }
                )
                SI.connect(p=p_I)
                brian_objects.append(SI)

            if C_E_ext > 0 and nu_ext_phys > 0.0:
                PGe = PoissonGroup(N_BRIAN, rates=(C_E_ext * nu_ext_phys) * hertz)
                SEext = Synapses(
                    PGe, G,
                    on_pre='v_post += J_E_ext_eff/Cb',
                    namespace={
                        'J_E_ext_eff': J_E_ext_eff,
                        'Cb': Cb
                    }
                )
                SEext.connect(j='i')
                brian_objects.extend([PGe, SEext])

            if C_I_ext > 0 and nu_ext_phys > 0.0:
                PGi = PoissonGroup(N_BRIAN, rates=(C_I_ext * nu_ext_phys) * hertz)
                SIext = Synapses(
                    PGi, G,
                    on_pre='v_post -= J_I_ext_eff/Cb',
                    namespace={
                        'J_I_ext_eff': J_I_ext_eff,
                        'Cb': Cb
                    }
                )
                SIext.connect(j='i')
                brian_objects.extend([PGi, SIext])

            spikemon = SpikeMonitor(G)
            brian_objects.append(spikemon)

            brian_times_mean_loc = []
            brian_mean_x_volt_loc = []

            @network_operation(dt=BRIAN_SAMPLE_DT * second, when='end')
            def collect_mean_x():
                brian_times_mean_loc.append(float(defaultclock.t / second))
                brian_mean_x_volt_loc.append(float(np.mean(G.x / volt)))

            brian_objects.append(collect_mean_x)

            net = Network(*brian_objects)
            net.run(T_phys * second,namespace={})

            # activity in 5 ms bins
            spike_times = np.asarray(spikemon.t / second, dtype=float)
            counts, _ = np.histogram(spike_times, bins=edges)
            activity_runs[run_id, :] = counts / (N_BRIAN * BIN_WIDTH)

            # mean voltage
            brian_times_mean_loc = np.asarray(brian_times_mean_loc, dtype=float)
            brian_mean_x_volt_loc = np.asarray(brian_mean_x_volt_loc, dtype=float)

            if brian_times_ref is None:
                brian_times_ref = brian_times_mean_loc.copy()

            Lx = min(mean_x_runs.shape[1], brian_mean_x_volt_loc.size)
            mean_x_runs[run_id, :Lx] = brian_mean_x_volt_loc[:Lx]
            if Lx < mean_x_runs.shape[1]:
                mean_x_runs[run_id, Lx:] = brian_mean_x_volt_loc[-1]

            # final density in x
            x_final_brian_volt = np.asarray(G.x / volt, dtype=float)

            x_volt = x_pde / U0_volt
            dx_volt = delta_x / U0_volt

            x_edges = np.empty(len(x_volt) + 1, dtype=float)
            x_edges[1:-1] = 0.5 * (x_volt[:-1] + x_volt[1:])
            x_edges[0] = x_volt[0] - 0.5 * dx_volt
            x_edges[-1] = x_volt[-1] + 0.5 * dx_volt

            rho_tmp, _ = np.histogram(
                x_final_brian_volt,
                bins=x_edges,
                density=True
            )
            rho_x_runs[run_id, :] = rho_tmp

            print(
                f"run {run_id+1}/{N_RUNS} done, "
                f"total spikes = {spike_times.size}, "
                f"max binned rate = {np.max(activity_runs[run_id, :]):.6e}"
            )

        t_brian1 = time.time()
        print(f"Brian ensemble done in {pretty_time(t_brian1 - t_brian0)}")

        mean_activity_brian = np.mean(activity_runs, axis=0)

        brian_times_mean = brian_times_ref.copy()
        mean_x_brian_interp = np.mean(mean_x_runs, axis=0)

        rho_x_brian_final = np.mean(rho_x_runs, axis=0)
        x_hist_centers = x_pde / U0_volt

    except Exception:
        print("[Brian2 disabled] Full traceback below:")
        traceback.print_exc()
        RUN_BRIAN = False

# Output folder

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = os.path.join("results", "pde-vs-brian2")
os.makedirs(outdir, exist_ok=True)

stem = (
    f"compare_pde_brian2_"
    f"T{T_phys}_n{n}_Nx{Nx}_Nv{Nv}_Nt{Nt}_{timestamp}"
)

# Graph 1: activity

t_cut = 0
mask_pde = times >= t_cut

fig1, ax1 = plt.subplots(figsize=(7, 4))
ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax1.plot(times[mask_pde], activities[mask_pde], '-', lw=1, label='PDE')

if RUN_BRIAN and (mean_activity_brian is not None):
    ax1.plot(tb_bin, mean_activity_brian, '--', lw=1,
             label='Brian 2')

ax1.set_xlabel(r"$t$ (s)")
ax1.set_ylabel("$N(t)$\n\n$\mathrm{(Hz)}$", rotation=0, labelpad=15)
ax1.set_title("Population activity")
ax1.legend(frameon=False)
fig1.tight_layout()

if SAVE_PDF:
    fig1.savefig(os.path.join(outdir, stem + "_activity.pdf"), bbox_inches='tight')
if SAVE_PNG:
    fig1.savefig(os.path.join(outdir, stem + "_activity.png"), dpi=DIAG_PLOT_DPI, bbox_inches='tight')
plt.close(fig1)

# Graph 2: mean voltage

fig2, ax2 = plt.subplots(figsize=(7, 4))
style_axes(ax2, yfmt="%.2f")
ax2.plot(times, mean_x_values / U0_volt, '-', lw=1, label='PDE')

if RUN_BRIAN and (mean_x_brian_interp is not None):
    Lm = min(len(brian_times_mean), len(mean_x_brian_interp))
    ax2.plot(brian_times_mean[:Lm], mean_x_brian_interp[:Lm], '--', lw=1.0,
             label='Brian 2')

ax2.set_xlabel(r"$t$ (s)")
ax2.set_ylabel("$X(t)$\n\n$\mathrm{(volt)}$", rotation=0, labelpad=15)
ax2.set_title("Mean voltage")
ax2.legend(frameon=False)
fig2.tight_layout()

if SAVE_PDF:
    fig2.savefig(os.path.join(outdir, stem + "_mean_voltage.pdf"), bbox_inches='tight')
if SAVE_PNG:
    fig2.savefig(os.path.join(outdir, stem + "_mean_voltage.png"), dpi=DIAG_PLOT_DPI, bbox_inches='tight')
plt.close(fig2)

# Graph 3: final x-density

fig3, ax3 = plt.subplots(figsize=(7, 4))
style_axes(ax3, yfmt="%.3f")

x_volt = x / U0_volt

rho_x_pde_final = delta_v * np.sum(f, axis=1)
rho_x_pde_final_volt = U0_volt * rho_x_pde_final

ax3.plot(x_volt, rho_x_pde_final_volt, '-', lw=1.0,
         label='PDE')

if RUN_BRIAN and (rho_x_brian_final is not None):
    ax3.plot(x_hist_centers, rho_x_brian_final, '--', lw=1.0,
             label='Brian 2')

ax3.set_xlabel(r"$x$ (volt)")
ax3.set_ylabel(r"$\rho(x,T)$", rotation=0, labelpad=20)
ax3.set_title("Final voltage density")
ax3.legend(frameon=False)
fig3.tight_layout()

if SAVE_PDF:
    fig3.savefig(os.path.join(outdir, stem + "_density_final.pdf"), bbox_inches='tight')
if SAVE_PNG:
    fig3.savefig(os.path.join(outdir, stem + "_density_final.png"), dpi=DIAG_PLOT_DPI, bbox_inches='tight')
plt.close(fig3)

# Final prints

print("\n")
print("Done")
print(f"Output folder: {os.path.abspath(outdir)}")
print(f"Mass initial   = {mass_of(f_initial):.16f}")
print(f"Mass final     = {mass_of(f):.16f}")
print(f"Min final f    = {np.min(f):.6e}")
if RUN_BRIAN and (mean_activity_brian is not None):
    print(f"Brian ensemble size = {N_RUNS}")
print("Saved graphs:")
print(" - activity")
print(" - mean voltage")
print(" - final density")