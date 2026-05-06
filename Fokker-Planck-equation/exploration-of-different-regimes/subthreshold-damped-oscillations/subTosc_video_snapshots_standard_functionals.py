# -*- coding: utf-8 -*-
"""
Vectorized tri banded solver and optional:
    - video f(x,v,t)
    - 9 equally-spaced snapshots as one 3x3 figure
    - mean voltage
    - activity
    - entropy
    - Fisher information

Created on Thu Apr 23 15:35:04 2026
@author: Nicolas Zadeh
"""

import os
import sys
import time
from datetime import datetime

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, PowerNorm

# Parameters

# Initial condition
x10 = 6
v10 = 20
sigma = 1

# Reset
u_F = 9
u_R = 4
sigma_rho = 0.001

# Grid
x_min = -31
x_max = u_F
size_x = x_max - x_min
X = max(u_F, abs(x_min))

v_min = -40
v_max = 40
size_v = v_max - v_min
V = max(abs(v_min), abs(v_max))

n = 20

# Electric parameters
b = 1
nu = 10
omega_0 = 0.5
tau = 5

a_0 = 0.5
a_1 = 0.5

if (1 / tau**2 - 4 * omega_0**2) < 0:
    print("Oscillatory framework")
else:
    raise RuntimeError("Non-oscillatory framework")

# Time parameters
T = 10
Nt = 50001
delta_t = np.float64(T / (Nt - 1))

# Colormap / normalization

colormap = "viridis"
powernorm = True
power_gamma = 0.5

# Visualization window
# To get proper scaling when the pdf point cloud doesn't get too spread out, comment
# out if necessary, I would advise first doing a trial run with small n 
# to see where the phenomenon happens then choosing a window adapted 
# to proper observation

DENSITY_XLIM = (-21, 9)
DENSITY_YLIM = (-5, 25)

# Clipped normalization
# To get proper scaling when the pdf values don't get too spread out, comment
# out if necessary, I would advise first doing a trial run with small n 
# to see where the phenomenon happens then choosing a window adapted 
# to proper observation
# Set CLIP_QUANTILE = None for no clipping.
CLIP_QUANTILE = 0.9999
CLIP_GAMMA = 0.5

if CLIP_QUANTILE is None:
    clip_tag = ""
else:
    clip_tag = f"_clip{100 * CLIP_QUANTILE:.2f}".replace(".", "p")

# Fonts, make them appear TeX-like

USE_TEX = True

mpl.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Visualization-only optional cosmetic smoothing

USE_GAUSSIAN_SMOOTHING = False
sigma_x_vis = 0.8
sigma_v_vis = 0.8

# Snapshot export parameters

SAVE_SNAPSHOTS = True
SAVE_INDIVIDUAL_SNAPSHOTS = False
SAVE_SNAPSHOT_GRID = True

NUM_SNAPSHOTS = 9
SNAPSHOT_DPI = 400

# Choice of saved plots

SAVE_ACTIVITY_PLOT = True
SAVE_ENTROPY_PLOT = True
SAVE_FISHER_PLOT = True
SAVE_EXPECTATION_PLOT = True
PLOT_DPI = 400

# Video export parameters

SAVE_VIDEO = True
target_duration = 20.0
fps = 30
VIDEO_DPI = 120

# Completion sounds

USE_SOUND = True

def play_beep(frequency=440.0, duration=0.3, volume=0.12):
    if not USE_SOUND:
        return

    if sys.platform.startswith("win"):
        import winsound
        winsound.Beep(int(frequency), int(1000 * duration))
        return

    path = None

    try:
        import shutil
        import subprocess
        import tempfile
        import wave

        fs = 44100
        n = int(fs * duration)
        t = np.linspace(0.0, duration, n, endpoint=False)

        signal = volume * np.sin(2.0 * np.pi * frequency * t)

        audio_mono = np.int16(signal * 32767)
        audio_stereo = np.column_stack([audio_mono, audio_mono]).ravel()
        # the temporary creation of a small wav file was the most stable 
        # multi-os solution
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name

        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(fs)
            w.writeframes(audio_stereo.tobytes())

        if sys.platform == "darwin":
            player = shutil.which("afplay")
            if player is None:
                raise RuntimeError("afplay not found")
            cmd = [player, path]

        elif sys.platform.startswith("linux"):
            player = shutil.which("paplay") or shutil.which("pw-play")
            if player is None:
                raise RuntimeError("Neither paplay nor pw-play was found")
            cmd = [player, path]

        else:
            raise RuntimeError(f"Unsupported platform for sound: {sys.platform}")

        subprocess.run(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

    except Exception as e:
        print(f"\nSound notification failed: {e}")

    finally:
        if path is not None:
            try:
                os.remove(path)
            except OSError:
                pass


def play_success_sound():
    play_beep(700, 0.2)
    time.sleep(0.005)
    play_beep(700, 0.1)
    time.sleep(0.05)
    play_beep(900, 0.8)


def play_failure_sound():
    play_beep(300, 0.2)
    time.sleep(0.05)
    play_beep(270, 0.2)
    time.sleep(0.05)
    play_beep(240, 0.2)
    time.sleep(0.05)
    play_beep(225, 0.8)


def safe_play_success_sound():
    try:
        play_success_sound()
    except Exception as e:
        print(f"\nSound notification failed, but simulation completed: {e}")


def safe_play_failure_sound():
    try:
        play_failure_sound()
    except Exception as e:
        print(f"\nFailure sound notification failed: {e}")

# Output folders

base_dir = os.path.dirname(os.path.abspath(__file__))

results_dir = os.path.join(base_dir, "results")
videos_dir = os.path.join(results_dir, "videos")
snapshots_dir = os.path.join(results_dir, "snapshots")
curves_dir = os.path.join(results_dir, "curves")

os.makedirs(videos_dir, exist_ok=True)
os.makedirs(snapshots_dir, exist_ok=True)
os.makedirs(curves_dir, exist_ok=True)

# Utilities

def fmt_float_for_filename(x):
    return f"{x:.2e}".replace(".", "p").replace("-", "m").replace("+", "")

def locate_ffmpeg():
    import shutil
    return shutil.which("ffmpeg")

def build_equally_spaced_indices(num_frames, num_selected):
    if num_selected <= 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, num_frames - 1, num_selected, dtype=int)
    return np.unique(idx)

def tex_num(x, digits=2):
    if abs(x) < 5e-15:
        x = 0.0
    return rf"${x:.{digits}f}$"

def tex_relevant_num(x, pos=None):
    if abs(x) < 5e-15:
        x = 0.0
    return rf"${x:g}$"

def tex_relevant_tick_formatter():
    return mticker.FuncFormatter(tex_relevant_num)

def apply_tex_ticks(ax):
    ax.xaxis.set_major_formatter(tex_relevant_tick_formatter())
    ax.yaxis.set_major_formatter(tex_relevant_tick_formatter())

def apply_tex_colorbar_ticks(cbar):
    cbar.ax.yaxis.set_major_formatter(tex_relevant_tick_formatter())
    cbar.update_ticks()

def tex_sci_num(x, pos=None):
    if abs(x) < 5e-15:
        return r"$0$"
    exp = int(np.floor(np.log10(abs(x))))
    mant = x / (10**exp)
    return rf"${mant:.2g}\times 10^{{{exp}}}$"

def tex_sci_tick_formatter():
    return mticker.FuncFormatter(tex_sci_num)

def tex_relevant_scaled_num(scale):
    def formatter(x, pos=None):
        z = scale * x
        if abs(z) < 5e-15:
            z = 0.0
        return rf"${z:g}$"
    return mticker.FuncFormatter(formatter)

def choose_activity_unit(y):
    """
    Choose a readable unit for activity values stored in Hz.

    The plotted data are not rescaled. Only the y tick labels are rescaled.
    """
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]

    if y.size == 0:
        return 1.0, r"Hz"

    ymax = np.max(np.abs(y))

    if ymax == 0.0:
        return 1.0, r"Hz"

    if ymax < 1e-9:
        return 1e12, r"pHz"
    if ymax < 1e-6:
        return 1e9, r"nHz"
    if ymax < 1e-3:
        return 1e6, r"$\mu$Hz"
    if ymax < 1.0:
        return 1e3, r"mHz"
    if ymax < 1e3:
        return 1.0, r"Hz"
    if ymax < 1e6:
        return 1e-3, r"kHz"

    return 1e-6, r"MHz"

def mass_of(f):
    return delta_x * delta_v * np.sum(f, dtype=np.float64)

def entropy_of(f):
    f_pos = f[f > 0.0]
    return np.float64(
        -delta_x * delta_v * np.sum(f_pos * np.log(f_pos), dtype=np.float64)
    )

def fisher_of(f):
    """
    Discrete Fisher information using the robust identity

        I(f) = 4 \int |\nabla sqrt(f)|^2 dx dv.
    """
    g = np.sqrt(np.maximum(f, 0.0))
    gx = np.gradient(g, delta_x, axis=0)
    gv = np.gradient(g, delta_v, axis=1)

    return np.float64(
        4.0 * delta_x * delta_v * np.sum(gx**2 + gv**2, dtype=np.float64)
    )

def mean_x_of(f):
    return np.float64(delta_x * delta_v * np.sum(x_col * f, dtype=np.float64))

def maybe_smooth(frame):
    if not USE_GAUSSIAN_SMOOTHING:
        return frame
    return gaussian_filter(
        frame,
        sigma=(sigma_x_vis, sigma_v_vis),
        mode="nearest"
    )

# Globals

Nx = Nv = None
delta_x = delta_v = None
dt_over_dx = dt_over_dv = None

x = v = None
x_col = v_row = None

f_initial = rho = None
i_F = j_0 = None

j_pos_x = j_neg_x = None
ab_row = None
J_full = None
x_interior_col = None
v_interior_row = None

# Initialization

def init_xv(n):
    global Nx, Nv, delta_x, delta_v, dt_over_dx, dt_over_dv
    global x, v, x_col, v_row
    global f_initial, rho, i_F, j_0
    global j_pos_x, j_neg_x
    global ab_row, J_full, x_interior_col, v_interior_row

    number_points_space = (n + 1) * size_x + 1
    number_points_velocity = (n + 1) * size_v + 1

    Nx = number_points_space - 2
    Nv = number_points_velocity - 2

    delta_x = np.float64(size_x / (Nx + 1))
    delta_v = np.float64(size_v / (Nv + 1))

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

    x_interior_col = x[1:i_F, None]
    v_interior_row = v[None, :]

    j_pos_x = slice(j_0 + 1, Nv + 2)
    j_neg_x = slice(0, j_0)

    inv2s2_init = 1.0 / (2.0 * sigma**2)
    gx_init = np.exp(-(x - x10)**2 * inv2s2_init)
    gv_init = np.exp(-(v - v10)**2 * inv2s2_init)
    f0 = np.multiply.outer(gx_init, gv_init)

    f0[0, j_0:] = 0.0
    f0[i_F, :j_0 + 1] = 0.0

    f_initial = f0 / (delta_x * delta_v * np.sum(f0, dtype=np.float64))
    f_initial *= 1.0 / (delta_x * delta_v * np.sum(f_initial, dtype=np.float64))

    print("Initial mass =", mass_of(f_initial))

    inv2s2_src = 1.0 / (2.0 * sigma_rho**2)
    gx_src = np.exp(-(x - u_R)**2 * inv2s2_src)
    gv_src = np.exp(-(v - 0.0)**2 * inv2s2_src)
    rho0 = np.multiply.outer(gx_src, gv_src)

    rho0[0, j_0:] = 0.0
    rho0[i_F, :j_0 + 1] = 0.0

    rho = rho0 / (delta_x * delta_v * np.sum(rho0, dtype=np.float64))

    print("Source mass =", mass_of(rho))

    ab_row = np.zeros((3, Nv + 2), dtype=np.float64)
    J_full = np.arange(Nv + 2, dtype=np.int64)[None, :]

# Implicit matrix in v

def build_ab_row(N):
    alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

    ab_row.fill(0.0)
    ab_row[1, :] = 1.0 + 2.0 * alpha
    ab_row[0, 1:] = -alpha
    ab_row[2, :-1] = -alpha

    ab_row[1, 0] = 1.0 + alpha
    ab_row[1, -1] = 1.0 + alpha

    return ab_row

# Transport operator

def apply_B_2d(f, N):
    out = f.copy()

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

    if i_F > 1:
        f_int = f[1:i_F, :]
        out_int = out[1:i_F, :]

        muv = -(omega_0**2) * x_interior_col - v_interior_row / tau + b * (nu + N)
        beta_v = -muv * dt_over_dv

        out_int += (delta_t / tau) * f_int

        jc = ((tau * (-omega_0**2 * x[1:i_F] + b * (nu + N)) + V) / delta_v).astype(np.int64)
        jc = np.clip(jc, -1, Nv + 1)

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

        mu_bottom = -(omega_0**2) * x[1:i_F] - v[0] / tau + b * (nu + N)
        mu_top = -(omega_0**2) * x[1:i_F] - v[Nv + 1] / tau + b * (nu + N)

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

# Time step

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

def build_common_norm(vmin, vmax):
    if powernorm:
        return PowerNorm(gamma=power_gamma, vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)

def build_clipped_power_norm_from_frames(frames, quantile=CLIP_QUANTILE, gamma=CLIP_GAMMA):
    vals = np.concatenate([frame.ravel() for frame in frames])

    if quantile is None:
        vmax_clip = np.max(vals)
    else:
        vmax_clip = np.quantile(vals, quantile)

    if vmax_clip <= 0:
        raise RuntimeError("The vmax is non-positive; cannot build a meaningful colormap.")

    return PowerNorm(gamma=gamma, vmin=0.0, vmax=vmax_clip), vmax_clip

def make_density_figure(frame2d, time_value, norm, minimalist_axes=False):
    fig, ax = plt.subplots(figsize=(6, 4))

    img = ax.imshow(
        frame2d.T,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], v[0], v[-1]],
        cmap=colormap,
        norm=norm,
        interpolation="nearest"
    )

    cbar = fig.colorbar(img, ax=ax)
    cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 7))
    cbar.set_label(r"$f$", rotation=0, labelpad=15)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.label.set_verticalalignment("center")
    apply_tex_colorbar_ticks(cbar)

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$", rotation=0, labelpad=15)
    ax.set_title(rf"$t = {time_value:g}\,$s")

    ax.set_xlim(*DENSITY_XLIM)
    ax.set_ylim(*DENSITY_YLIM)

    if minimalist_axes:
        ax.set_xticks([u_R, u_F])
        ax.set_xticklabels([r"$u_{\rm R}$", r"$u_{\rm F}$"])
        ax.set_yticks([0])
        ax.set_yticklabels([r"$0$"])
    else:
        apply_tex_ticks(ax)

    fig.subplots_adjust(
        left=0.14,
        right=0.86,
        bottom=0.16,
        top=0.90
    )

    return fig, ax, img

def save_snapshot(frame2d, time_value, idx, outdir, stem, norm):
    fig, ax, img = make_density_figure(frame2d, time_value, norm)

    base = os.path.join(outdir, f"{stem}_snapshot_{idx+1:02d}_t{time_value:.3f}")
    pdf_name = base + ".pdf"
    fig.savefig(pdf_name, dpi=SNAPSHOT_DPI)

    plt.close(fig)

def save_snapshot_grid(frames, times_snap, outdir, stem, norm):
    fig, axes = plt.subplots(
        3, 3,
        figsize=(9.5, 8.2),
        sharex=True,
        sharey=True
    )

    last_img = None

    for k, (ax, frame, tval) in enumerate(zip(axes.flat, frames, times_snap)):
        last_img = ax.imshow(
            frame.T,
            origin="lower",
            aspect="auto",
            extent=[x[0], x[-1], v[0], v[-1]],
            cmap=colormap,
            norm=norm,
            interpolation="nearest"
        )

        ax.set_xlim(*DENSITY_XLIM)
        ax.set_ylim(*DENSITY_YLIM)

        ax.set_title(rf"$t={tval:g}\,$s", fontsize=11)

        row = k // 3
        col = k % 3

        if row == 2:
            ax.set_xlabel(r"$x$")
            ax.set_xticks([u_R, u_F])
            ax.set_xticklabels([r"$u_{\rm R}$", r"$u_{\rm F}$"])
            ax.tick_params(axis="x", labelbottom=True)
        else:
            ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

        if col == 0:
            ax.set_ylabel(r"$v$", rotation=0, labelpad=12)
            ax.set_yticks([0, 10, 20])
            ax.tick_params(axis="y", labelleft=True)
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        bottom=0.08,
        top=0.93,
        wspace=0.08,
        hspace=0.20
    )

    bottom = axes[-1, 0].get_position().y0
    top = axes[0, 0].get_position().y1

    cbar_ax = fig.add_axes([0.90, bottom, 0.025, top - bottom])
    cbar = fig.colorbar(last_img, cax=cbar_ax)
    cbar.set_label(r"$f$", rotation=0, labelpad=18, fontsize=16)
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.label.set_verticalalignment("center")
    apply_tex_colorbar_ticks(cbar)

    grid_filename = os.path.join(
        outdir,
        stem + f"_3x3_snapshots{clip_tag}_window_uR_uF.pdf"
    )

    fig.savefig(grid_filename, dpi=SNAPSHOT_DPI)
    plt.close(fig)

    return grid_filename

def save_curve_plot(t, y, xlabel, ylabel, title, filename_base,
                    sci_y="auto", hide_yticks=False,
                    adaptive_activity_units=False,
                    base_activity_label=r"$N(t)$"):
    fig = plt.figure(figsize=(6, 4))

    # Fixed axes box for all curve plots:
    # [left, bottom, width, height] in figure coordinates.
    ax = fig.add_axes([0.13, 0.16, 0.84, 0.74])

    ax.plot(t, y, linewidth=1.8)

    ax.set_xlabel(xlabel)

    if title is not None:
        ax.set_title(title)

    ax.xaxis.set_major_formatter(tex_relevant_tick_formatter())

    y_finite = np.asarray(y, dtype=np.float64)
    y_finite = y_finite[np.isfinite(y_finite)]

    if adaptive_activity_units:
        scale, unit = choose_activity_unit(y_finite)

        ax.yaxis.get_offset_text().set_visible(False)
        ax.yaxis.set_major_formatter(tex_relevant_scaled_num(scale))

        ylabel_final = base_activity_label + "\n\n" + rf"({unit})"

    else:
        if y_finite.size == 0:
            use_scientific = False
        elif sci_y is True:
            use_scientific = True
        elif sci_y is False:
            use_scientific = False
        else:
            ymax = np.max(np.abs(y_finite))
            use_scientific = (ymax >= 1e3) or (0 < ymax < 1e-2)

        if use_scientific:
            ax.yaxis.set_major_formatter(tex_sci_tick_formatter())
        else:
            ax.yaxis.set_major_formatter(tex_relevant_tick_formatter())

        ylabel_final = ylabel

    if hide_yticks:
        ax.set_yticks([])
        ax.tick_params(axis="y", which="both",
                       left=False, right=False, labelleft=False)

    ax.set_ylabel(ylabel_final, rotation=0, labelpad=20)

    pdf_name = filename_base + ".pdf"

    # Important: do NOT use bbox_inches="tight".
    fig.savefig(pdf_name, dpi=PLOT_DPI)

    plt.close(fig)

# Main

try:

    np.set_printoptions(precision=25)

    start_time = time.time()
    init_xv(n)

    print(f"Nx = {Nx}, Nv = {Nv}, dx = {delta_x}, dv = {delta_v}, dt = {delta_t}")

    # Filename stem

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = (
        f"f_xv_"
        f"Nx{Nx}_Nv{Nv}_"
        f"T{T}_Nt{Nt}_"
        f"{timestamp}"
    )
    
    ffmpeg_path = locate_ffmpeg() if SAVE_VIDEO else None
    VIDEO_AVAILABLE = SAVE_VIDEO and (ffmpeg_path is not None)
    
    if SAVE_VIDEO and not VIDEO_AVAILABLE:
        print("\nWARNING: ffmpeg was not found in PATH.")
        print("To enable video export, install ffmpeg and ensure it is in PATH.")
        print("Video frames will not be stored.")
        print("Snapshots and diagnostic plots will still be saved.")

    video_filename = os.path.join(videos_dir, file_stem + ".mp4")
    diag_base = os.path.join(curves_dir, file_stem)

    # Precompute selected indices

    snapshot_indices = build_equally_spaced_indices(Nt, NUM_SNAPSHOTS) if SAVE_SNAPSHOTS else np.array([], dtype=int)

    n_video_frames = max(2, int(round(target_duration * fps))) if VIDEO_AVAILABLE else 0
    video_indices = build_equally_spaced_indices(Nt, n_video_frames) if VIDEO_AVAILABLE else np.array([], dtype=int)

    selected_indices = np.unique(np.concatenate([snapshot_indices, video_indices]))
    selected_set = set(int(k) for k in selected_indices)

    print("Snapshot indices =", snapshot_indices.tolist())

    # Allocate functionals of interest only

    times = np.linspace(0.0, T, Nt, dtype=np.float64)
    activities = np.empty(Nt, dtype=np.float64)
    masses = np.empty(Nt, dtype=np.float64)
    entropies = np.empty(Nt, dtype=np.float64)
    fishers = np.empty(Nt, dtype=np.float64)
    mean_x_values = np.empty(Nt, dtype=np.float64)

    # Store only selected frames
    stored_frames = {}

    # Initial state

    f = f_initial.copy()
    f_plot = maybe_smooth(f)
    vmax_global = np.max(f_plot)

    activities[0] = compute_activity(f)
    masses[0] = mass_of(f)
    entropies[0] = entropy_of(f)
    fishers[0] = fisher_of(f)
    mean_x_values[0] = mean_x_of(f)

    if 0 in selected_set:
        stored_frames[0] = f_plot.copy()

    # Time loop

    for k in range(1, Nt):
        f, N = step(f)
        f_plot = maybe_smooth(f)

        activities[k] = N
        masses[k] = mass_of(f)
        entropies[k] = entropy_of(f)
        fishers[k] = fisher_of(f)
        mean_x_values[k] = mean_x_of(f)

        frame_max = np.max(f_plot)
        if frame_max > vmax_global:
            vmax_global = frame_max

        if k in selected_set:
            stored_frames[k] = f_plot.copy()

        if (k % 10000 == 0) or (k == Nt - 1):
            print(
                f"k={k}/{Nt-1}, t={times[k]:.4f}, "
                f"N={activities[k]:.6e}, mass={masses[k]:.16f}, "
                f"S={entropies[k]:.16f}, I={fishers[k]:.16f}, "
                f"min(f)={np.min(f):.6e}"
            )

    print(f"Simulation done in {time.time() - start_time:.2f} s")

    # Save snapshots / snapshot grid

    if SAVE_SNAPSHOTS:
        print("Snapshot times   =", [float(times[idx]) for idx in snapshot_indices])

        snapshot_frames = [stored_frames[int(idx)] for idx in snapshot_indices]
        snapshot_times = [times[int(idx)] for idx in snapshot_indices]

        snapshot_norm, snapshot_vmax_clip = build_clipped_power_norm_from_frames(
            snapshot_frames,
            quantile=CLIP_QUANTILE,
            gamma=CLIP_GAMMA
        )

        if CLIP_QUANTILE is None:
            print("Snapshot norm: no clipping, vmax =", snapshot_vmax_clip)
        else:
            print(f"Snapshot norm: clip {100 * CLIP_QUANTILE:.2f}% vmax =", snapshot_vmax_clip)

        if SAVE_INDIVIDUAL_SNAPSHOTS:
            for k_snap, idx in enumerate(snapshot_indices):
                save_snapshot(
                    frame2d=stored_frames[int(idx)],
                    time_value=times[int(idx)],
                    idx=k_snap,
                    outdir=snapshots_dir,
                    stem=file_stem + clip_tag,
                    norm=snapshot_norm
                )

        if SAVE_SNAPSHOT_GRID:
            grid_filename = save_snapshot_grid(
                frames=snapshot_frames,
                times_snap=snapshot_times,
                outdir=snapshots_dir,
                stem=file_stem,
                norm=snapshot_norm
            )

            print("Saved snapshot grid to:")
            print(os.path.abspath(grid_filename))

    # Save functionals of interest

    if SAVE_ACTIVITY_PLOT:
        save_curve_plot(
            t=times,
            y=activities,
            xlabel=r"$t$ (s)",
            ylabel=r"$N(t)$" + "\n\n" + r"(Hz)",
            title=None,
            filename_base=diag_base + "_activity",
            sci_y=False,
            adaptive_activity_units=True,
            base_activity_label=r"$N(t)$"
        )

    if SAVE_ENTROPY_PLOT:
        save_curve_plot(
            t=times,
            y=entropies,
            xlabel=r"$t$ (s)",
            ylabel=r"$H(f)(t)$",
            title=None,
            filename_base=diag_base + "_entropy",
            sci_y=False,
            hide_yticks=True
        )

    if SAVE_FISHER_PLOT:
        save_curve_plot(
            t=times,
            y=fishers,
            xlabel=r"$t$ (s)",
            ylabel=r"$I(f)(t)$",
            title=None,
            filename_base=diag_base + "_fisher",
            sci_y="auto",
            hide_yticks=True
        )

    if SAVE_EXPECTATION_PLOT:
        save_curve_plot(
            t=times,
            y=mean_x_values,
            xlabel=r"$t$ (s)",
            ylabel=r"$X(t)$" + "\n\n" + r"(volt)",
            title=None,
            filename_base=diag_base + "_mean_x",
            sci_y=False
        )

    # Save video from stored selected frames only

    if VIDEO_AVAILABLE:       
        mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        print("Using ffmpeg at:", ffmpeg_path)

        frames_video = np.array([stored_frames[int(idx)] for idx in video_indices], dtype=np.float64)
        times_video = np.array([times[int(idx)] for idx in video_indices], dtype=np.float64)

        video_norm, video_vmax_clip = build_clipped_power_norm_from_frames(
            frames_video,
            quantile=CLIP_QUANTILE,
            gamma=CLIP_GAMMA
        )

        if CLIP_QUANTILE is None:
            print("Video norm: no clipping, vmax =", video_vmax_clip)
        else:
            print(f"Video norm: clip {100 * CLIP_QUANTILE:.2f}% vmax =", video_vmax_clip)

        writer = animation.FFMpegWriter(
            fps=fps,
            codec="libx264",
            bitrate=-1,
            extra_args=[
                "-preset", "ultrafast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
            ]
        )

        # Standard video

        fig, ax, img = make_density_figure(
            frame2d=frames_video[0],
            time_value=times_video[0],
            norm=video_norm,
            minimalist_axes=False
        )

        def update(frame_idx):
            img.set_data(frames_video[frame_idx].T)
            ax.set_title(rf"$t = {times_video[frame_idx]:g}\,$s")
            return (img,)

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames_video),
            interval=1000 / fps,
            blit=False
        )

        video_filename_clip = video_filename.replace(".mp4", f"{clip_tag}.mp4")
        ani.save(video_filename_clip, writer=writer, dpi=VIDEO_DPI)

        print("Video saved to:", os.path.abspath(video_filename_clip))
        print(f"Chosen fps = {fps}, duration of approximately {len(frames_video) / fps:.3f} s")

        plt.close(fig)

        # Minimalist axes video:
        # x-axis: only u_R and u_F
        # v-axis: only 0

        fig_min, ax_min, img_min = make_density_figure(
            frame2d=frames_video[0],
            time_value=times_video[0],
            norm=video_norm,
            minimalist_axes=True
        )

        def update_minimalist_axes(frame_idx):
            img_min.set_data(frames_video[frame_idx].T)
            ax_min.set_title(rf"$t = {times_video[frame_idx]:g}\,$s")
            return (img_min,)

            def update_minimalist_axes(frame_idx):
                img_min.set_data(frames_video[frame_idx].T)
                ax_min.set_title(rf"$t = {times_video[frame_idx]:g}\,$s")
                return (img_min,)

            ani_min = animation.FuncAnimation(
                fig_min,
                update_minimalist_axes,
                frames=len(frames_video),
                interval=1000 / fps,
                blit=False
            )

            video_filename_minimalist = video_filename.replace(
                ".mp4",
                f"{clip_tag}_minimal_axes_uR_uF_v0.mp4"
            )

            ani_min.save(video_filename_minimalist, writer=writer, dpi=VIDEO_DPI)

            print("Minimalist axes video saved to:", os.path.abspath(video_filename_minimalist))

            plt.close(fig_min)

except Exception as e:
    print("Simulation failed:", e)
    safe_play_failure_sound()
    raise

else:
    print("Simulation completed successfully")
    safe_play_success_sound()
