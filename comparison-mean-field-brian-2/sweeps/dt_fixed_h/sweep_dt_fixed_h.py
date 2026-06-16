# -*- coding: utf-8 -*-
"""
The script performs a numerical time-step sweep at fixed h=Delta x=Delta v.

Outputs:
- for each value of dt: comparison plots for N(t), X(t), rho(x,T);
- a CSV table of L2 distances against the Brian ensemble mean;
- log-log plots of dt versus L2 distance.

The relative values are obtained by divisions against the Brian 2 norm.

Nicolas Zadeh, June the 13th, 2026
"""

import csv
import math
import os
import sys
import time
import traceback
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.linalg import solve_banded

warnings.filterwarnings("ignore", category=UserWarning)

# User choices

RUN_BRIAN = True

# If True, Brian2 is run once per J, using the largest left domain as the
# lower truncation for the initial condition. The final Brian positions are
# then re-binned on each PDE x-grid. This is much cheaper.
#
# If False, Brian2 is run separately for each pair (x_min,J).
REUSE_BRIAN_ACROSS_SWEEP = True

SAVE_PDF = True
SAVE_PNG = False
DIAG_PLOT_DPI = 350

PLOT_BRIAN_STD = True
BRIAN_STD_ALPHA = 0.20

# Fixed physical values from the original program
X_MIN_FIXED = -9.0
K_FIXED = 1.0e-5       # original J=1e-4 with C=1, tau_syn=10

# Numerical sweep values
# The target mesh size h is fixed. We only vary Delta t.
# This value corresponds to the old N_FIXED = 20 through h = 1/(20+1).
H_FIXED = 1.0 / 21.0
DT_VALUES = [5.0e-4, 1.0e-4, 5.0e-5, 1.0e-5, 5.0e-6]

# Brian2 is run once with this reference time step and then interpolated on
# the PDE grids.
BRIAN_REFERENCE_DT = 1.0e-5

# Time horizon
T_phys = 1.0
T = T_phys

# Target mesh size.
h = H_FIXED

# The associated grid level is still recorded for information.
n = int(round(1.0 / h - 1.0))

# PDE time steps
Nt_user = int(round(T / DT_VALUES[0])) + 1

# Brian parameters
N_RUNS = 10
NE = 80000
NI = 20000
N_BRIAN = NE + NI

# Brian activity bin width
BRIAN_RATE_BIN_MS = 5.0
BIN_WIDTH = BRIAN_RATE_BIN_MS * 1.0e-3

# Regime parameters

U0_volt = 1.0

# Threshold and reset
u_F = 9.0
u_R = 4.0

# Velocity domain, fixed while x_min varies
V_DOMAIN = 25.0

# Initial condition, kept identical throughout the sweep
x10 = 4.0
v10 = 15.0
sigma_init_x = 3.0 * U0_volt
sigma_init_v = 3.0 * U0_volt

# Reset/source regularization
sigma_rho_x = 0.001 * U0_volt
sigma_rho_v = 0.001 * U0_volt

# Physical parameters
R_phys = 0.8
R_L_phys = 0.8
C_phys = 1.0
L_phys = 2.0
tau_syn_phys = 10.0

# External rate
nu_ext_phys = 0.1

# Connectivity
C_E_ext = 800
C_I_ext = 200
C_E_int = 800
C_I_int = 200

DTYPE = np.float64

# Formatting utilities

def pretty_time(s):
    m, s = divmod(float(s), 60.0)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:06.3f}"


def style_axes(ax, yfmt="%.2f"):
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter(yfmt))
    ax.tick_params(
        axis='both',
        which='both',
        direction='out',
        length=2,
        width=0.8,
    )


def set_horizontal_ylabel(ax, label, labelpad=70):
    ax.set_ylabel(label, rotation=0, labelpad=labelpad, ha='right', va='center')


def format_yaxis_single_power(ax):
    formatter = mtick.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(formatter)


def fmt_float_for_filename(value):
    return f"{value:.2e}".replace('.', 'p').replace('-', 'm').replace('+', '')


def fmt_xmin_for_filename(value):
    return f"{value:.0f}".replace('-', 'm')


def l2_distance_on_grid(a, b, grid):
    diff2 = np.asarray(a - b, dtype=np.float64) ** 2
    return float(np.sqrt(np.trapz(diff2, x=grid)))


def relative_l2_distance_on_grid(a, b, grid, eps=1.0e-300):
    numerator = l2_distance_on_grid(a, b, grid)
    denominator = float(
        np.sqrt(np.trapz(np.asarray(b, dtype=np.float64) ** 2, x=grid))
    )
    return numerator / max(denominator, eps)


def sample_truncated_normal(rng, mean, sigma, low, high, size):
    samples = np.empty(size, dtype=float)
    filled = 0

    while filled < size:
        remaining = size - filled
        candidates = rng.normal(
            loc=mean,
            scale=sigma,
            size=max(2 * remaining, 1024),
        )
        candidates = candidates[(candidates >= low) & (candidates <= high)]

        take = min(remaining, candidates.size)
        if take > 0:
            samples[filled:filled + take] = candidates[:take]
            filled += take

    return samples


# Global physical coefficients, updated when the kick size changes

kick_size = None
J = None
J_E_ext = J_I_ext = J_E_int = J_I_int = None

omega0_phys = None
tau_phys = None
beta_phys = None
omega_0 = None
tau = None

nu = None
nu_ext = None
b_ext = None
b_int = None
a_0 = None
a_1 = None


def update_kick_dependent_parameters(kick_value):
    global kick_size, J, J_E_ext, J_I_ext, J_E_int, J_I_int
    global omega0_phys, tau_phys, beta_phys, omega_0, tau
    global nu, nu_ext, b_ext, b_int, a_0, a_1

    kick_size = float(kick_value)

    # J is the microscopic synaptic area.
    # The actual velocity kick size is J/(C tau_syn), hence
    # J = kick_size * C * tau_syn.
    J = kick_size * C_phys * tau_syn_phys

    J_E_ext = J
    J_I_ext = J
    J_E_int = J
    J_I_int = J

    omega0_phys = math.sqrt(
        (R_phys + R_L_phys) / (R_phys * L_phys * C_phys)
    )
    tau_phys = (
        R_phys * L_phys * C_phys
    ) / (L_phys + R_L_phys * R_phys * C_phys)
    beta_phys = omega0_phys * tau_phys

    omega_0 = omega0_phys
    tau = tau_phys

    b_ext = (
        C_E_ext * J_E_ext - C_I_ext * J_I_ext
    ) / (C_phys * tau_syn_phys)
    b_int = (
        C_E_int * J_E_int - C_I_int * J_I_int
    ) / (C_phys * tau_syn_phys)

    a_0 = (
        (C_E_ext * J_E_ext**2 + C_I_ext * J_I_ext**2)
        / (2.0 * C_phys**2 * tau_syn_phys**2)
    ) * nu_ext_phys

    a_1 = (
        C_E_int * J_E_int**2 + C_I_int * J_I_int**2
    ) / (2.0 * C_phys**2 * tau_syn_phys**2)

    nu = nu_ext = nu_ext_phys


# Global domain and grid variables

x_min = None
x_max = u_F
v_min = None
v_max = None
size_x = None
size_v = None
X = None
V = None

Nx = Nv = None
delta_x = delta_v = None
delta_t = None
dt_over_dx = dt_over_dv = None

x = v = None
x_col = v_row = None

f_initial = None
rho = None
i_F = j_0 = None

j_pos_x = j_neg_x = None
interior_rows = None

ab_row = None
J_full = None
x_interior_col = None
v_interior_row = None


def update_domain_parameters(x_min_value):
    global x_min, x_max, v_min, v_max, size_x, size_v, X, V

    x_min = float(x_min_value)
    x_max = float(u_F)

    v_min = -float(V_DOMAIN) * U0_volt
    v_max = float(V_DOMAIN) * U0_volt

    size_x = x_max - x_min
    size_v = v_max - v_min

    X = max(abs(x_min), abs(x_max))
    V = max(abs(v_min), abs(v_max))


# PDE utilities

def mass_of(f):
    return np.float64(delta_x * delta_v * np.sum(f, dtype=np.float64))


def mean_x_of(f):
    return np.float64(
        delta_x * delta_v * np.sum(x_col * f, dtype=np.float64)
    )


def init_xv(h_value):
    global Nx, Nv, delta_x, delta_v, delta_t, dt_over_dx, dt_over_dv
    global x, v, x_col, v_row
    global f_initial, rho, i_F, j_0
    global j_pos_x, j_neg_x, interior_rows
    global ab_row, J_full, x_interior_col, v_interior_row

    target_h = float(h_value)

    n_intervals_x = int(round(size_x / target_h))
    n_intervals_v = int(round(size_v / target_h))

    Nx = n_intervals_x - 1
    Nv = n_intervals_v - 1

    delta_x = np.float64(size_x / n_intervals_x)
    delta_v = np.float64(size_v / n_intervals_v)

    if not np.isclose(delta_x, delta_v, rtol=1e-13, atol=1e-15):
        raise ValueError(
            "dx and dv are not equal. Choose compatible domain lengths "
            f"or change n. Got dx={delta_x}, dv={delta_v}."
        )

    delta_t = np.float64(T / (Nt_user - 1))
    dt_over_dx = delta_t / delta_x
    dt_over_dv = delta_t / delta_v

    x = np.linspace(x_min, x_max, Nx + 2, dtype=np.float64)
    v = np.linspace(v_min, v_max, Nv + 2, dtype=np.float64)

    i_F = Nx + 1
    j_0 = int(round((0.0 - v_min) / delta_v))

    if abs(x[i_F] - u_F) > 1.0e-12:
        raise RuntimeError("u_F is not on the x-grid.")
    if abs(v[j_0]) > 1.0e-12:
        raise RuntimeError("0 is not on the v-grid.")

    x_col = x[:, None]
    v_row = v[None, :]

    interior_rows = slice(1, i_F)
    x_interior_col = x[1:i_F, None]
    v_interior_row = v[None, :]

    j_pos_x = slice(j_0 + 1, Nv + 2)
    j_neg_x = slice(0, j_0)

    # Initial condition
    inv2s2_init_x = 1.0 / (2.0 * sigma_init_x**2)
    gx_init = np.exp(-(x - x10)**2 * inv2s2_init_x)

    inv2s2_init_v = 1.0 / (2.0 * sigma_init_v**2)
    gv_init = np.exp(-(v - v10)**2 * inv2s2_init_v)

    f0 = np.multiply.outer(gx_init, gv_init)

    f0[0, j_0:] = 0.0
    f0[i_F, :j_0 + 1] = 0.0

    f_initial = f0 / mass_of(f0)

    # Source
    inv2s2_src_x = 1.0 / (2.0 * sigma_rho_x**2)
    gx_src = np.exp(-(x - u_R)**2 * inv2s2_src_x)

    inv2s2_src_v = 1.0 / (2.0 * sigma_rho_v**2)
    gv_src = np.exp(-(v - 0.0)**2 * inv2s2_src_v)

    rho0 = np.multiply.outer(gx_src, gv_src)

    rho0[0, j_0:] = 0.0
    rho0[i_F, :j_0 + 1] = 0.0

    rho_mass = mass_of(rho0)
    if rho_mass <= 0.0:
        raise RuntimeError(
            "The reset source has zero numerical mass. "
            "Check that u_R and v=0 lie on the grid."
        )

    rho = rho0 / rho_mass

    # Row-wise implicit matrix A
    ab_row = np.zeros((3, Nv + 2), dtype=np.float64)

    # j-array reused in critical-strip construction
    J_full = np.arange(Nv + 2, dtype=np.int64)[None, :]


def build_ab_row(N_value):
    alpha = (a_0 + a_1 * N_value) * delta_t / (delta_v**2)

    ab_row.fill(0.0)

    ab_row[1, :] = 1.0 + 2.0 * alpha
    ab_row[0, 1:] = -alpha
    ab_row[2, :-1] = -alpha

    ab_row[1, 0] = 1.0 + alpha
    ab_row[1, -1] = 1.0 + alpha

    return ab_row


def apply_B_2d(f, N_value):
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

        muv = (
            -omega_0**2 * x_interior_col
            - v_interior_row / tau
            + b_ext * nu_ext
            + b_int * N_value
        )
        beta_v = -muv * dt_over_dv

        out_int += (delta_t / tau) * f_int

        jc = (
            (
                tau * (-omega_0**2 * x[1:i_F] + b_ext * nu_ext + b_int * N_value)
                + V
            ) / delta_v
        ).astype(np.int64)

        crit_mask = (J_full == jc[:, None]) | (J_full == (jc[:, None] + 1))

        out_int += np.where(
            crit_mask & (muv > 0.0),
            -(beta_v + delta_t / tau) * f_int,
            0.0,
        )

        out_int += np.where(
            crit_mask & (muv < 0.0),
            (beta_v - delta_t / tau) * f_int,
            0.0,
        )

        if Nv >= 1:
            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] > 0.0,
                beta_v[:, 1:Nv + 1] * (
                    f_int[:, 1:Nv + 1] - f_int[:, 0:Nv]
                ),
                0.0,
            )

            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] < 0.0,
                beta_v[:, 1:Nv + 1] * (
                    f_int[:, 2:Nv + 2] - f_int[:, 1:Nv + 1]
                ),
                0.0,
            )

        mu_bottom = (
            -omega_0**2 * x[1:i_F]
            - v[0] / tau
            + b_ext * nu_ext
            + b_int * N_value
        )
        mu_top = (
            -omega_0**2 * x[1:i_F]
            - v[Nv + 1] / tau
            + b_ext * nu_ext
            + b_int * N_value
        )

        out[1:i_F, 0] += -mu_bottom * dt_over_dv * f[1:i_F, 0]
        out[1:i_F, Nv + 1] += mu_top * dt_over_dv * f[1:i_F, Nv + 1]

    return out


def solve_A_rowwise(D2, N_value):
    fnew = np.empty_like(D2)

    fnew[0, :] = D2[0, :]
    fnew[i_F, :] = D2[i_F, :]

    if i_F > 1:
        ab = build_ab_row(N_value)
        rhs = D2[1:i_F, :].T
        sol = solve_banded((1, 1), ab, rhs)
        fnew[1:i_F, :] = sol.T

    return fnew


def compute_activity(f):
    return np.float64(
        delta_v * np.sum(
            f[i_F, j_0 + 1:Nv + 2] * v[j_0 + 1:Nv + 2],
            dtype=np.float64,
        )
        - delta_v * np.sum(
            f[0, 0:j_0] * v[0:j_0],
            dtype=np.float64,
        )
    )


def step(f):
    N_value = compute_activity(f)
    D2 = apply_B_2d(f, N_value) + N_value * delta_t * rho
    fnew = solve_A_rowwise(D2, N_value)
    return fnew, N_value


# PDE run globals

pde_times = None
pde_activities = None
pde_mean_x_values = None
pde_rho_x_final = None
pde_f_final = None

pde_mass_initial = None
pde_mass_final = None
pde_min_final = None


def run_pde_one_case(dt_value):
    global pde_times, pde_activities, pde_mean_x_values
    global pde_rho_x_final, pde_f_final
    global pde_mass_initial, pde_mass_final, pde_min_final
    global h, n, Nt_user

    h = float(H_FIXED)
    n = int(round(1.0 / h - 1.0))
    Nt_user = int(round(T / dt_value)) + 1

    update_kick_dependent_parameters(K_FIXED)
    update_domain_parameters(X_MIN_FIXED)
    init_xv(h)

    print(
        f"\n[PDE] h = {h:.12g}, n_equiv = {n}, dt = {delta_t:.6e}, "
        f"x_min = {x_min:g}, K = {kick_size:.1e}, "
        f"Nx = {Nx}, Nv = {Nv}, dx = {delta_x}, dv = {delta_v}"
    )

    f = f_initial.copy()

    pde_times = np.empty(Nt_user, dtype=np.float64)
    pde_activities = np.empty(Nt_user, dtype=np.float64)
    pde_mean_x_values = np.empty(Nt_user, dtype=np.float64)

    pde_times[0] = 0.0
    pde_mean_x_values[0] = mean_x_of(f) / U0_volt
    pde_mass_initial = mass_of(f)

    t0 = time.time()

    for k in range(1, Nt_user):
        f, N_k_minus_1 = step(f)

        current_time = k * delta_t
        current_mass = mass_of(f)
        current_mean_x = mean_x_of(f)

        pde_times[k] = current_time
        pde_activities[k - 1] = N_k_minus_1
        pde_mean_x_values[k] = current_mean_x / U0_volt

        if abs(current_mass - 1.0) > 1e-1:
            print(f"At moment k={k}, mass={current_mass:.16f}")
            sys.exit("mass conservation failed")

        if (k % 1000 == 0) or (k == Nt_user - 1):
            print(
                f"    k={k}/{Nt_user-1}, t={current_time:.4f}, "
                f"N={N_k_minus_1:.6e}, "
                f"mass={current_mass:.16f}, "
                f"mean={current_mean_x / U0_volt:.16f}"
            )

    pde_activities[-1] = compute_activity(f)

    pde_f_final = f
    pde_rho_x_final = U0_volt * delta_v * np.sum(f, axis=1)
    pde_mass_final = mass_of(f)
    pde_min_final = float(np.min(f))

    print(f"[PDE] done in {pretty_time(time.time() - t0)}")
    print(f"    mass initial = {pde_mass_initial:.16f}")
    print(f"    mass final   = {pde_mass_final:.16f}")
    print(f"    min final f  = {pde_min_final:.6e}")

# Brian2 raw ensemble globals and current-grid summaries

brian_tb_bin_raw = None
brian_activity_runs_raw = None
brian_times_mean_raw = None
brian_mean_x_runs_raw = None
brian_x_final_runs_raw = None

mean_activity_brian = None
std_activity_brian = None
tb_bin = None

mean_x_brian_interp = None
std_mean_x_brian_interp = None
brian_times_mean = None

rho_x_brian_final = None
std_rho_x_brian_final = None
x_hist_centers = None


def run_brian_ensemble(kick_value, init_x_min):
    global brian_tb_bin_raw, brian_activity_runs_raw
    global brian_times_mean_raw, brian_mean_x_runs_raw
    global brian_x_final_runs_raw

    update_kick_dependent_parameters(kick_value)

    print(
        f"\n[Brian2] K = {kick_size:.1e}, "
        f"J_area = {J:.6e}, init lower x = {init_x_min:g}"
    )

    try:
        from brian2 import (  # type: ignore
            prefs, start_scope, defaultclock,
            second, volt, amp, ohm, farad, hertz,
            NeuronGroup, Synapses, SpikeMonitor,
            PoissonGroup, Network, network_operation,
        )

        if N_BRIAN <= 0:
            raise ValueError("N_BRIAN must be > 0 to run Brian2.")

        prefs.codegen.target = "numpy"

        brian_dt = float(BRIAN_REFERENCE_DT)

        edges = np.arange(0.0, T_phys + BIN_WIDTH, BIN_WIDTH)
        if edges[-1] < T_phys:
            edges = np.append(edges, T_phys)
        brian_tb_bin_raw = edges[:-1]

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

        brian_activity_runs_raw = np.empty((N_RUNS, len(brian_tb_bin_raw)), dtype=float)
        brian_Nt = int(round(T_phys / brian_dt)) + 1
        brian_mean_x_runs_raw = np.empty((N_RUNS, brian_Nt), dtype=float)
        brian_x_final_runs_raw = np.empty((N_RUNS, N_BRIAN), dtype=float)
        brian_times_ref = None

        t0 = time.time()

        for run_id in range(N_RUNS):
            start_scope()
            defaultclock.dt = brian_dt * second

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
                },
            )

            x0_brian = sample_truncated_normal(
                rng=rng,
                mean=x10,
                sigma=sigma_init_x,
                low=init_x_min,
                high=np.nextafter(u_F, -np.inf),
                size=N_BRIAN,
            )

            v0_brian = sample_truncated_normal(
                rng=rng,
                mean=v10,
                sigma=sigma_init_v,
                low=-V_DOMAIN,
                high=V_DOMAIN,
                size=N_BRIAN,
            )

            G.x = x0_brian * volt
            G.v = v0_brian * volt / second
            G.I = 0.0 * amp

            brian_objects = [G]

            Ge = G[:NE] if NE > 0 else None
            Gi = G[NE:NE + NI] if NI > 0 else None

            if NE > 0:
                p_E = min(1.0, C_E_int / NE)
                SE = Synapses(
                    Ge,
                    G,
                    on_pre='v_post += J_E_int_eff/Cb',
                    namespace={
                        'J_E_int_eff': J_E_int_eff,
                        'Cb': Cb,
                    },
                )
                SE.connect(p=p_E)
                brian_objects.append(SE)

            if NI > 0:
                p_I = min(1.0, C_I_int / NI)
                SI = Synapses(
                    Gi,
                    G,
                    on_pre='v_post -= J_I_int_eff/Cb',
                    namespace={
                        'J_I_int_eff': J_I_int_eff,
                        'Cb': Cb,
                    },
                )
                SI.connect(p=p_I)
                brian_objects.append(SI)

            if C_E_ext > 0 and nu_ext_phys > 0.0:
                PGe = PoissonGroup(
                    N_BRIAN,
                    rates=(C_E_ext * nu_ext_phys) * hertz,
                )
                SEext = Synapses(
                    PGe,
                    G,
                    on_pre='v_post += J_E_ext_eff/Cb',
                    namespace={
                        'J_E_ext_eff': J_E_ext_eff,
                        'Cb': Cb,
                    },
                )
                SEext.connect(j='i')
                brian_objects.extend([PGe, SEext])

            if C_I_ext > 0 and nu_ext_phys > 0.0:
                PGi = PoissonGroup(
                    N_BRIAN,
                    rates=(C_I_ext * nu_ext_phys) * hertz,
                )
                SIext = Synapses(
                    PGi,
                    G,
                    on_pre='v_post -= J_I_ext_eff/Cb',
                    namespace={
                        'J_I_ext_eff': J_I_ext_eff,
                        'Cb': Cb,
                    },
                )
                SIext.connect(j='i')
                brian_objects.extend([PGi, SIext])

            spikemon = SpikeMonitor(G)
            brian_objects.append(spikemon)

            brian_times_mean_loc = []
            brian_mean_x_volt_loc = []

            @network_operation(dt=brian_dt * second, when='end')
            def collect_mean_x():
                brian_times_mean_loc.append(float(defaultclock.t / second))
                brian_mean_x_volt_loc.append(float(np.mean(G.x / volt)))

            brian_objects.append(collect_mean_x)

            net = Network(*brian_objects)
            net.run(T_phys * second, namespace={})

            spike_times = np.asarray(spikemon.t / second, dtype=float)
            counts, _ = np.histogram(spike_times, bins=edges)
            brian_activity_runs_raw[run_id, :] = counts / (N_BRIAN * BIN_WIDTH)

            brian_times_mean_loc = np.asarray(brian_times_mean_loc, dtype=float)
            brian_mean_x_volt_loc = np.asarray(brian_mean_x_volt_loc, dtype=float)

            if brian_times_ref is None:
                brian_times_ref = brian_times_mean_loc.copy()

            Lx = min(brian_mean_x_runs_raw.shape[1], brian_mean_x_volt_loc.size)
            brian_mean_x_runs_raw[run_id, :Lx] = brian_mean_x_volt_loc[:Lx]
            if Lx < brian_mean_x_runs_raw.shape[1]:
                brian_mean_x_runs_raw[run_id, Lx:] = brian_mean_x_volt_loc[-1]

            brian_x_final_runs_raw[run_id, :] = np.asarray(G.x / volt, dtype=float)

            print(
                f"    run {run_id+1}/{N_RUNS} done, "
                f"total spikes = {spike_times.size}, "
                f"max binned rate = {np.max(brian_activity_runs_raw[run_id, :]):.6e}"
            )

        if brian_times_ref is None:
            raise RuntimeError("Brian2 produced no time samples.")

        brian_times_mean_raw = brian_times_ref.copy()

        print(f"[Brian2] done in {pretty_time(time.time() - t0)}")
        return True

    except Exception:
        print("[Brian2 disabled] Full traceback below:")
        traceback.print_exc()

        brian_tb_bin_raw = None
        brian_activity_runs_raw = None
        brian_times_mean_raw = None
        brian_mean_x_runs_raw = None
        brian_x_final_runs_raw = None

        return False


def build_brian_summary_on_current_grid():
    global mean_activity_brian, std_activity_brian, tb_bin
    global mean_x_brian_interp, std_mean_x_brian_interp, brian_times_mean
    global rho_x_brian_final, std_rho_x_brian_final, x_hist_centers

    if brian_x_final_runs_raw is None:
        return False

    x_edges = np.empty(len(x) + 1, dtype=float)
    x_edges[1:-1] = 0.5 * (x[:-1] + x[1:]) / U0_volt
    x_edges[0] = x[0] / U0_volt - 0.5 * delta_x / U0_volt
    x_edges[-1] = x[-1] / U0_volt + 0.5 * delta_x / U0_volt

    rho_x_runs = np.empty((N_RUNS, len(x)), dtype=float)

    for run_id in range(N_RUNS):
        rho_tmp, _ = np.histogram(
            brian_x_final_runs_raw[run_id, :],
            bins=x_edges,
            density=True,
        )
        rho_x_runs[run_id, :] = rho_tmp

    tb_bin = brian_tb_bin_raw.copy()
    brian_times_mean = brian_times_mean_raw.copy()

    mean_activity_brian = np.mean(brian_activity_runs_raw, axis=0)
    mean_x_brian_interp = np.mean(brian_mean_x_runs_raw, axis=0)
    rho_x_brian_final = np.mean(rho_x_runs, axis=0)

    if N_RUNS > 1:
        std_activity_brian = np.std(brian_activity_runs_raw, axis=0, ddof=1)
        std_mean_x_brian_interp = np.std(brian_mean_x_runs_raw, axis=0, ddof=1)
        std_rho_x_brian_final = np.std(rho_x_runs, axis=0, ddof=1)
    else:
        std_activity_brian = np.zeros_like(mean_activity_brian)
        std_mean_x_brian_interp = np.zeros_like(mean_x_brian_interp)
        std_rho_x_brian_final = np.zeros_like(rho_x_brian_final)

    x_hist_centers = x / U0_volt

    return True


# Plots and distances

def plot_comparison_curves(outdir, stem, brian_available):
    # Activity
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax1.plot(pde_times, pde_activities, '-', lw=1.0, label='PDE')

    if brian_available and mean_activity_brian is not None:
        line_brian, = ax1.plot(
            tb_bin,
            mean_activity_brian,
            '--',
            lw=1.0,
            label='Brian 2 mean',
        )

        if PLOT_BRIAN_STD and N_RUNS > 1:
            ax1.fill_between(
                tb_bin,
                mean_activity_brian - std_activity_brian,
                mean_activity_brian + std_activity_brian,
                color=line_brian.get_color(),
                alpha=BRIAN_STD_ALPHA,
                linewidth=0,
                label=r'Brian 2 $\pm$ std.',
            )

    ax1.set_xlabel(r"$t$ (s)")
    ax1.set_ylabel("$N(t)$\n\n$\\mathrm{(Hz)}$", rotation=0, labelpad=15)
    ax1.set_title(
        rf"Population activity, $x_{{\min}}={x_min:g}$, "
        rf"$K={kick_size:.0e}$"
    )
    ax1.legend(frameon=False)
    fig1.tight_layout()

    if SAVE_PDF:
        fig1.savefig(os.path.join(outdir, stem + "_activity.pdf"), bbox_inches='tight')
    if SAVE_PNG:
        fig1.savefig(
            os.path.join(outdir, stem + "_activity.png"),
            dpi=DIAG_PLOT_DPI,
            bbox_inches='tight',
        )
    plt.close(fig1)

    # Mean voltage
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    style_axes(ax2, yfmt="%.2f")
    ax2.plot(pde_times, pde_mean_x_values, '-', lw=1.0, label='PDE')

    if brian_available and mean_x_brian_interp is not None:
        Lm = min(len(brian_times_mean), len(mean_x_brian_interp))
        line_brian_x, = ax2.plot(
            brian_times_mean[:Lm],
            mean_x_brian_interp[:Lm],
            '--',
            lw=1.0,
            label='Brian 2 mean',
        )

        if PLOT_BRIAN_STD and N_RUNS > 1:
            ax2.fill_between(
                brian_times_mean[:Lm],
                mean_x_brian_interp[:Lm] - std_mean_x_brian_interp[:Lm],
                mean_x_brian_interp[:Lm] + std_mean_x_brian_interp[:Lm],
                color=line_brian_x.get_color(),
                alpha=BRIAN_STD_ALPHA,
                linewidth=0,
                label=r'Brian 2 $\pm$ std.',
            )

    ax2.set_xlabel(r"$t$ (s)")
    ax2.set_ylabel("$X(t)$\n\n$\\mathrm{(volt)}$", rotation=0, labelpad=15)
    ax2.set_title(
        rf"Mean voltage, $x_{{\min}}={x_min:g}$, "
        rf"$K={kick_size:.0e}$"
    )
    ax2.legend(frameon=False)
    fig2.tight_layout()

    if SAVE_PDF:
        fig2.savefig(os.path.join(outdir, stem + "_mean_voltage.pdf"), bbox_inches='tight')
    if SAVE_PNG:
        fig2.savefig(
            os.path.join(outdir, stem + "_mean_voltage.png"),
            dpi=DIAG_PLOT_DPI,
            bbox_inches='tight',
        )
    plt.close(fig2)

    # Final density
    fig3, ax3 = plt.subplots(figsize=(7, 4))
    style_axes(ax3, yfmt="%.3f")
    x_volt = x / U0_volt
    ax3.plot(x_volt, pde_rho_x_final, '-', lw=1.0, label='PDE')

    if brian_available and rho_x_brian_final is not None:
        line_brian_rho, = ax3.plot(
            x_hist_centers,
            rho_x_brian_final,
            '--',
            lw=1.0,
            label='Brian 2 mean',
        )

        if PLOT_BRIAN_STD and N_RUNS > 1:
            ax3.fill_between(
                x_hist_centers,
                rho_x_brian_final - std_rho_x_brian_final,
                rho_x_brian_final + std_rho_x_brian_final,
                color=line_brian_rho.get_color(),
                alpha=BRIAN_STD_ALPHA,
                linewidth=0,
                label=r'Brian 2 $\pm$ std.',
            )

    ax3.set_xlabel(r"$x$ (volt)")
    ax3.set_ylabel(r"$\rho(x,T)$", rotation=0, labelpad=20)
    ax3.set_title(
        rf"Final voltage density, $x_{{\min}}={x_min:g}$, "
        rf"$K={kick_size:.0e}$"
    )
    ax3.legend(frameon=False)
    fig3.tight_layout()

    if SAVE_PDF:
        fig3.savefig(os.path.join(outdir, stem + "_density_final.pdf"), bbox_inches='tight')
    if SAVE_PNG:
        fig3.savefig(
            os.path.join(outdir, stem + "_density_final.png"),
            dpi=DIAG_PLOT_DPI,
            bbox_inches='tight',
        )
    plt.close(fig3)


def compute_distances(brian_available):
    if not brian_available:
        return (float('nan'), float('nan'), float('nan'),
                float('nan'), float('nan'), float('nan'))

    brian_activity_on_pde = np.interp(
        pde_times,
        tb_bin,
        mean_activity_brian,
        left=mean_activity_brian[0],
        right=mean_activity_brian[-1],
    )

    Lm = min(len(brian_times_mean), len(mean_x_brian_interp))
    brian_x_on_pde = np.interp(
        pde_times,
        brian_times_mean[:Lm],
        mean_x_brian_interp[:Lm],
        left=mean_x_brian_interp[0],
        right=mean_x_brian_interp[Lm - 1],
    )

    L2_N = l2_distance_on_grid(pde_activities, brian_activity_on_pde, pde_times)
    L2_X = l2_distance_on_grid(pde_mean_x_values, brian_x_on_pde, pde_times)
    L2_rho = l2_distance_on_grid(
        pde_rho_x_final,
        rho_x_brian_final,
        x / U0_volt,
    )

    rel_L2_N = relative_l2_distance_on_grid(
        pde_activities,
        brian_activity_on_pde,
        pde_times,
    )
    rel_L2_X = relative_l2_distance_on_grid(
        pde_mean_x_values,
        brian_x_on_pde,
        pde_times,
    )
    rel_L2_rho = relative_l2_distance_on_grid(
        pde_rho_x_final,
        rho_x_brian_final,
        x / U0_volt,
    )

    return L2_N, L2_X, L2_rho, rel_L2_N, rel_L2_X, rel_L2_rho


# CSV and summary plots

CSV_HEADER = [
    'h',
    'n_equiv',
    'dt',
    'dx',
    'dv',
    'Nt',
    'Nx',
    'Nv',
    'x_min',
    'K',
    'J_area',
    'mass_initial',
    'mass_final',
    'min_final_f',
    'L2_N',
    'L2_X',
    'L2_rho',
    'rel_L2_N',
    'rel_L2_X',
    'rel_L2_rho',
]


def save_records_csv(records, outdir):
    csv_path = os.path.join(outdir, "sweep_L2_distances.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(CSV_HEADER)
        for row in records:
            writer.writerow(row)

    return csv_path


def get_record_value(row, key):
    return row[CSV_HEADER.index(key)]


def plot_summary(records, outdir):
    if len(records) == 0:
        return

    observables = [
        ('N', 'L2_N', r"$\|N_{PDE}-N_{Brian}\|_{L^2(0,T)}$"),
        ('X', 'L2_X', r"$\|X_{PDE}-X_{Brian}\|_{L^2(0,T)}$"),
        ('rho', 'L2_rho', r"$\|\rho_{PDE}(\cdot,T)-\rho_{Brian}(\cdot,T)\|_{L^2_x}$"),
    ]

    dt_values = sorted(set(float(get_record_value(row, 'dt')) for row in records))

    def value_at(dt_value, key):
        for row in records:
            if np.isclose(float(get_record_value(row, 'dt')), dt_value):
                return float(get_record_value(row, key))
        return float('nan')

    for obs_name, key, ylabel in observables:
        yvals = np.array([value_at(dt_value, key) for dt_value in dt_values])

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.loglog(dt_values, yvals, marker='o', lw=1.0)
        ax.set_xlabel(r"$\Delta t$")
        set_horizontal_ylabel(ax, ylabel)
        ax.set_title(rf"Variation with $\Delta t$: {obs_name}, $h={H_FIXED:.3g}$")
        fig.tight_layout()
        fig.subplots_adjust(left=0.30)

        if SAVE_PDF:
            fig.savefig(
                os.path.join(outdir, f"summary_loglog_dt_L2_{obs_name}.pdf"),
                bbox_inches='tight',
            )
        if SAVE_PNG:
            fig.savefig(
                os.path.join(outdir, f"summary_loglog_dt_L2_{obs_name}.png"),
                dpi=DIAG_PLOT_DPI,
                bbox_inches='tight',
            )
        plt.close(fig)

# Main sweep

def print_global_parameters():
    update_kick_dependent_parameters(K_FIXED)

    print("PHYSICAL PARAMETERS FROM ORIGINAL PROGRAM")
    print(f"omega0_phys = {omega0_phys:.6g} s^-1")
    print(f"tau_phys    = {tau_phys:.6g} s")
    print(f"beta_phys   = {beta_phys:.6g}")
    print()
    print(f"x_min fixed = {X_MIN_FIXED:g}")
    print(f"K fixed     = {K_FIXED:.6e}")
    print(f"J_area      = {J:.6e}")
    print(f"T           = {T:.6e} s")
    print(f"H_FIXED      = {H_FIXED:.12g}")
    print(f"n_equiv      = {n}")
    print(f"DT_VALUES     = {DT_VALUES}")
    print(f"BRIAN_REFERENCE_DT = {BRIAN_REFERENCE_DT:.6e}")
    print(f"N_RUNS = {N_RUNS}, N_BRIAN = {N_BRIAN}")
    print(f"REUSE_BRIAN_ACROSS_SWEEP = {REUSE_BRIAN_ACROSS_SWEEP}")
    print()

    if (1.0 / tau**2 - 4.0 * omega_0**2) < 0.0:
        print("Oscillatory regime (underdamped).")
    else:
        print("Warning: not in oscillatory regime.")


def main():
    np.set_printoptions(precision=25)
    print_global_parameters()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(
        "results",
        "pde-vs-brian2-dt-sweep-fixed-h",
        timestamp,
    )
    os.makedirs(outdir, exist_ok=True)

    records = []
    t_all = time.time()

    brian_available_for_sweep = False
    if RUN_BRIAN and REUSE_BRIAN_ACROSS_SWEEP:
        brian_available_for_sweep = run_brian_ensemble(K_FIXED, X_MIN_FIXED)

    for dt_value in DT_VALUES:
        run_pde_one_case(dt_value)

        brian_available = False
        if RUN_BRIAN:
            if REUSE_BRIAN_ACROSS_SWEEP:
                brian_available = brian_available_for_sweep
            else:
                brian_available = run_brian_ensemble(K_FIXED, X_MIN_FIXED)

            if brian_available:
                brian_available = build_brian_summary_on_current_grid()

        stem = (
            f"compare_dt{fmt_float_for_filename(delta_t)}_"
            f"h{fmt_float_for_filename(h)}_Nx{Nx}_Nv{Nv}_Nt{Nt_user}"
        )

        plot_comparison_curves(outdir, stem, brian_available)

        (
            L2_N,
            L2_X,
            L2_rho,
            rel_L2_N,
            rel_L2_X,
            rel_L2_rho,
        ) = compute_distances(brian_available)

        row = [
            h,
            n,
            delta_t,
            delta_x,
            delta_v,
            Nt_user,
            Nx,
            Nv,
            x_min,
            kick_size,
            J,
            pde_mass_initial,
            pde_mass_final,
            pde_min_final,
            L2_N,
            L2_X,
            L2_rho,
            rel_L2_N,
            rel_L2_X,
            rel_L2_rho,
        ]
        records.append(row)

        print("\nDistances against Brian ensemble mean:")
        print(f"    L2_N   = {L2_N:.8e}")
        print(f"    L2_X   = {L2_X:.8e}")
        print(f"    L2_rho = {L2_rho:.8e}")

        csv_path = save_records_csv(records, outdir)
        plot_summary(records, outdir)
        print(f"    Updated CSV: {os.path.abspath(csv_path)}")

    csv_path = save_records_csv(records, outdir)
    plot_summary(records, outdir)

    print("\n")
    print("Done")
    print(f"Total wall time: {pretty_time(time.time() - t_all)}")
    print(f"Output folder: {os.path.abspath(outdir)}")
    print(f"Distance table: {os.path.abspath(csv_path)}")
    print("Saved per-parameter comparison graphs:")
    print(" - activity")
    print(" - mean voltage")
    print(" - final density")
    print("Saved summary graphs:")
    print(" - log-log dt versus L2 distance")

if __name__ == "__main__":
    main()