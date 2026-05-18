# -*- coding: utf-8 -*-
"""
3D video of the density f(x,v,t) produced by the 
semi-implicit upwind scheme in the reset-driven regime.

Nicolas Zadeh, May the 15th, 2026.
"""

import os
import time
from datetime import datetime

import numpy as np
from scipy.linalg import solve_banded
from scipy.ndimage import gaussian_filter

import matplotlib as mpl
# I don't want the Qt interactive video,
# I don't encourage its use here
mpl.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from matplotlib.colors import Normalize, PowerNorm

# Parameters

# Initial condition
x10 = -10
v10 = -5
sigma = 2

# Reset
u_F = 9
u_R = 4
sigma_rho = 0.001

# Grid
x_min = -23
x_max = u_F
size_x = x_max - x_min
X = max(u_F, abs(x_min))

v_min = -30
v_max = 30
size_v = v_max - v_min
V = max(abs(v_min), abs(v_max))

# n = 5
n = 14

# Electric parameters
b = 1
nu = 1
omega_0 = 1
tau = 1

a_0 = 1
a_1 = 1

if (1 / tau**2 - 4 * omega_0**2) < 0:
    print("Oscillatory framework")
else:
    raise RuntimeError("Non-oscillatory framework")

# Time parameters
T = 8.0
Nt = 50001
delta_t = np.float64(T / (Nt - 1))

# Visualization parameters

colormap = "viridis"
powernorm = True
power_gamma = 0.5
TIME_TITLE_DIGITS = 2

# Dynamic window, inherited from the 2D transient video script.
PLOT_XLIM = (-18, 9)
PLOT_VLIM = (-15, 12)

# Optional clipping of the 3D height/color scale.
# Use None for true max. 
Z_CLIP_QUANTILE = None

# View requested by the user.
VIEW_ELEV = 35
VIEW_AZIM = 45

# Manual 3D video layout.
FIGSIZE_3D = (9.5, 6.2)
AX_POSITION = [0.10, 0.07, 0.70, 0.84]
COLORBAR_POSITION = [0.84, 0.24, 0.025, 0.48]
SHOW_COLORBAR = True
SHOW_Z_TICK_LABELS = True

# Fonts.
USE_TEX = True

mpl.rcParams.update({
    "text.usetex": USE_TEX,
    "font.family": "serif",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

# Visualization-only optional smoothing.
USE_GAUSSIAN_SMOOTHING = False
sigma_x_vis = 0.8
sigma_v_vis = 0.8

# Video export parameters.
SAVE_VIDEO = True
target_duration = 20.0
fps = 30
VIDEO_DPI = 120

# Output folders

base_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(base_dir, "results")
videos_dir = os.path.join(results_dir, "videos")
os.makedirs(videos_dir, exist_ok=True)

# Utilities

def locate_ffmpeg():
    import shutil
    return shutil.which("ffmpeg")


def build_equally_spaced_indices(num_frames, num_selected):
    if num_selected <= 1:
        return np.array([0], dtype=int)
    idx = np.linspace(0, num_frames - 1, num_selected, dtype=int)
    return np.unique(idx)


def tex_relevant_num(x, pos=None):
    if abs(x) < 5e-15:
        x = 0.0
    return rf"${x:g}$"


def tex_relevant_tick_formatter():
    return mticker.FuncFormatter(tex_relevant_num)


def apply_tex_colorbar_ticks(cbar):
    cbar.ax.yaxis.set_major_formatter(tex_relevant_tick_formatter())
    cbar.update_ticks()


def maybe_smooth(frame):
    if not USE_GAUSSIAN_SMOOTHING:
        return frame
    return gaussian_filter(
        frame,
        sigma=(sigma_x_vis, sigma_v_vis),
        mode="nearest"
    )


def build_common_norm(vmin, vmax):
    if powernorm:
        return PowerNorm(
            gamma=power_gamma,
            vmin=vmin,
            vmax=vmax,
            clip=True
        )
    return Normalize(vmin=vmin, vmax=vmax, clip=True)


def polish_3d_axes(ax, elev, azim):
    """
    Axis cosmetics for the 3D video, the 
    first naive tries were too blunt
    """
    ax.tick_params(axis="x", pad=2)
    ax.tick_params(axis="y", pad=2)
    ax.tick_params(axis="z", pad=3)

    ax.set_xlabel(r"$x$", labelpad=10)
    ax.set_ylabel(r"$v$", labelpad=10)

    ax.zaxis.set_rotate_label(False)
    # The colorbar already says "f" 
    ax.set_zlabel("")

def safe_view_init(ax, elev, azim):
    try:
        ax.view_init(elev=elev, azim=azim, roll=0)
    except TypeError:
        ax.view_init(elev=elev, azim=azim)
        
def format_time_title(t):
    return rf"$t = {t:.{TIME_TITLE_DIGITS}f}\,$s"

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

# 3D rendering grid globals
x_surface = v_surface = None
X3 = V3 = None
surface_ix = surface_iv = None

# Functionals

def mass_of(f):
    return delta_x * delta_v * np.sum(f, dtype=np.float64)

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


def init_surface_grid():
    """
    Build the fixed plotting grid for the 3D animation.
    """
    global x_surface, v_surface, X3, V3, surface_ix, surface_iv

    ix_all = np.where((x >= PLOT_XLIM[0]) & (x <= PLOT_XLIM[1]))[0]
    iv_all = np.where((v >= PLOT_VLIM[0]) & (v <= PLOT_VLIM[1]))[0]

    if ix_all.size == 0 or iv_all.size == 0:
        raise RuntimeError("The requested 3D plotting window is empty.")

    surface_ix = ix_all
    surface_iv = iv_all

    x_surface = x[surface_ix]
    v_surface = v[surface_iv]
    X3, V3 = np.meshgrid(x_surface, v_surface, indexing="ij")

def extract_surface_frame(f):
    return f[np.ix_(surface_ix, surface_iv)].copy()

# Numerical scheme

def build_ab_row(N):
    alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

    ab_row.fill(0.0)
    ab_row[1, :] = 1.0 + 2.0 * alpha
    ab_row[0, 1:] = -alpha
    ab_row[2, :-1] = -alpha

    ab_row[1, 0] = 1.0 + alpha
    ab_row[1, -1] = 1.0 + alpha

    return ab_row

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

        muv = (
            -(omega_0**2) * x_interior_col
            - v_interior_row / tau
            + b * (nu + N)
        )
        beta_v = -muv * dt_over_dv

        out_int += (delta_t / tau) * f_int

        jc = (
            (
                tau * (-(omega_0**2) * x[1:i_F] + b * (nu + N))
                + V
            )
            / delta_v
        ).astype(np.int64)

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
                beta_v[:, 1:Nv + 1]
                * (f_int[:, 1:Nv + 1] - f_int[:, 0:Nv]),
                0.0
            )

            out_int[:, 1:Nv + 1] += np.where(
                muv[:, 1:Nv + 1] < 0.0,
                beta_v[:, 1:Nv + 1]
                * (f_int[:, 2:Nv + 2] - f_int[:, 1:Nv + 1]),
                0.0
            )

        mu_bottom = (
            -(omega_0**2) * x[1:i_F]
            - v[0] / tau
            + b * (nu + N)
        )
        mu_top = (
            -(omega_0**2) * x[1:i_F]
            - v[Nv + 1] / tau
            + b * (nu + N)
        )

        out[1:i_F, 0] += -mu_bottom * dt_over_dv * f[1:i_F, 0]
        out[1:i_F, Nv + 1] += mu_top * dt_over_dv * f[1:i_F, Nv + 1]

    return out


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
        delta_v
        * np.sum(
            f[i_F, j_0 + 1:Nv + 2] * v[j_0 + 1:Nv + 2],
            dtype=np.float64
        )
        - delta_v
        * np.sum(
            f[0, 0:j_0] * v[0:j_0],
            dtype=np.float64
        )
    )


def step(f):
    N = compute_activity(f)
    D2 = apply_B_2d(f, N) + N * delta_t * rho
    fnew = solve_A_rowwise(D2, N)
    return fnew, N

# 3D video

def compute_zmax(frames):
    vals = np.concatenate([frame.ravel() for frame in frames])

    if Z_CLIP_QUANTILE is None:
        zmax = np.max(vals)
    else:
        zmax = np.quantile(vals, Z_CLIP_QUANTILE)

    if zmax <= 0.0:
        raise RuntimeError("The plotted density is non-positive; cannot build z-scale.")

    return np.float64(zmax)


def make_3d_figure(first_frame, first_time, zmax, norm):
    fig = plt.figure(figsize=FIGSIZE_3D, constrained_layout=False)

    ax = fig.add_axes(AX_POSITION, projection="3d")

    Z = np.minimum(first_frame, zmax)

    surf = ax.plot_surface(
        X3, V3, Z,
        cmap=colormap,
        norm=norm,
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1
    )

    safe_view_init(ax, elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.set_xlim(*PLOT_XLIM)
    ax.set_ylim(*PLOT_VLIM)
    ax.set_zlim(0.0, zmax)

    polish_3d_axes(ax, elev=VIEW_ELEV, azim=VIEW_AZIM)

    ax.set_xticks([u_R, u_F])
    ax.set_xticklabels([r"$u_{\rm R}$", r"$u_{\rm F}$"])

    ax.set_yticks([-10.0, 0.0, 10.0])
    ax.set_yticklabels([r"$-10$", r"$0$", r"$10$"])

    zticks = np.linspace(0.0, zmax, 5)
    ax.set_zticks(zticks)
    ax.zaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    if not SHOW_Z_TICK_LABELS:
        ax.set_zticklabels([])

    ax.set_title(format_time_title(first_time), pad=10)

    # Improve the centering of the figure
    try:
        ax.set_box_aspect((
            PLOT_XLIM[1] - PLOT_XLIM[0],
            PLOT_VLIM[1] - PLOT_VLIM[0],
            0.55 * max(
                PLOT_XLIM[1] - PLOT_XLIM[0],
                PLOT_VLIM[1] - PLOT_VLIM[0],
            ),
        ))
    except AttributeError:
        pass

    if SHOW_COLORBAR:
        cbar_ax = fig.add_axes(COLORBAR_POSITION)
        cbar = fig.colorbar(surf, cax=cbar_ax)
        cbar.set_label(r"$f$", rotation=0, labelpad=18)
        cbar.ax.yaxis.set_label_position("right")
        cbar.ax.yaxis.label.set_verticalalignment("center")
        # Fixed colorbar ticks, avoiding an extra tick above zmax.
        cbar_ticks = np.linspace(0.0, zmax, 5)
        cbar.set_ticks(cbar_ticks)
        cbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    return fig, ax, surf

def save_3d_video(frames_video, times_video, video_filename):
    ffmpeg_path = locate_ffmpeg()
    if ffmpeg_path is None:
        raise RuntimeError(
            "ffmpeg was not found in PATH. Install ffmpeg before saving MP4 videos."
        )

    mpl.rcParams["animation.ffmpeg_path"] = ffmpeg_path
    print("Using ffmpeg at:", ffmpeg_path)

    zmax = compute_zmax(frames_video)
    norm = build_common_norm(vmin=0.0, vmax=zmax)

    if Z_CLIP_QUANTILE is None:
        print("3D z-scale: no clipping, zmax =", zmax)
    else:
        print(f"3D z-scale: clip {100 * Z_CLIP_QUANTILE:.3f}% zmax =", zmax)

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

    fig, ax, surf = make_3d_figure(
        first_frame=frames_video[0],
        first_time=times_video[0],
        zmax=zmax,
        norm=norm
    )

    current_surface = [surf]

    def update(frame_idx):
        current_surface[0].remove()

        Z = np.minimum(frames_video[frame_idx], zmax)
        current_surface[0] = ax.plot_surface(
            X3, V3, Z,
            cmap=colormap,
            norm=norm,
            linewidth=0,
            antialiased=True,
            rstride=1,
            cstride=1
        )

        ax.set_title(format_time_title(times_video[frame_idx]))
        safe_view_init(ax, elev=VIEW_ELEV, azim=VIEW_AZIM)

        return (current_surface[0],)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=len(frames_video),
        interval=1000 / fps,
        blit=False
    )

    ani.save(video_filename, writer=writer, dpi=VIDEO_DPI)

    print("3D video saved to:", os.path.abspath(video_filename))
    print(f"Chosen fps = {fps}, duration = {len(frames_video) / fps:.3f} s")

    plt.close(fig)

# Main

def main():
    np.set_printoptions(precision=25)

    start_time = time.time()

    init_xv(n)
    init_surface_grid()

    print(f"Nx = {Nx}, Nv = {Nv}")
    print(f"dx = {delta_x}, dv = {delta_v}, dt = {delta_t}")
    print(f"T = {T}, Nt = {Nt}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_stem = (
        f"f_xv_3D_"
        f"Nx{Nx}_Nv{Nv}_"
        f"T{T}_Nt{Nt}_"
        f"elev{VIEW_ELEV}_azim{VIEW_AZIM}_"
        f"{timestamp}"
    )

    video_filename = os.path.join(
        videos_dir,
        file_stem + ".mp4"
    )

    n_video_frames = max(2, int(round(target_duration * fps)))
    video_indices = build_equally_spaced_indices(Nt, n_video_frames)
    selected_set = set(int(k) for k in video_indices)

    print("Video frame count =", len(video_indices))
    print("First/last video indices =", int(video_indices[0]), int(video_indices[-1]))

    times = np.linspace(0.0, T, Nt, dtype=np.float64)
    stored_frames = {}

    f = f_initial.copy()

    if 0 in selected_set:
        stored_frames[0] = extract_surface_frame(maybe_smooth(f))

    for k in range(1, Nt):
        f, N_k_minus_1 = step(f)

        if k in selected_set:
            stored_frames[k] = extract_surface_frame(maybe_smooth(f))

        if (k % 5000 == 0) or (k == Nt - 1):
            print(
                f"k={k}/{Nt-1}, t={times[k]:.4f}, "
                f"N_k_minus_1={N_k_minus_1:.6e}, mass={mass_of(f):.16f}, "
                f"min(f)={np.min(f):.6e}, max(f)={np.max(f):.6e}"
            )

    print(f"Simulation done in {time.time() - start_time:.2f} s")

    frames_video = [stored_frames[int(idx)] for idx in video_indices]
    times_video = np.array([times[int(idx)] for idx in video_indices], dtype=np.float64)

    if SAVE_VIDEO:
        save_3d_video(
            frames_video=frames_video,
            times_video=times_video,
            video_filename=video_filename
        )

    print("Program completed successfully")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Program failed:", e)
        raise
