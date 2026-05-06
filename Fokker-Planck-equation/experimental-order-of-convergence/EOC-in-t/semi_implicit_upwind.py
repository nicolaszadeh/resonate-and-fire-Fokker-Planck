# -*- coding: utf-8 -*-
"""
Semi-implicit upwind-in-transport:
- diffusion in v implicit at time n+1,
- transport explicit at time n,
- special critical strip j_c, j_c+1,
- row-wise banded solver in v,
- experimental order of convergence in time (fixed space grid),
- CSV saving of results.
- if a floating-point overflow / invalid / divide-by-zero happens
  during one run, that run is aborted immediately,
- the code then continues with the next Nt,
- failed runs are stored as nan in the CSV.
"""

import numpy as np
import time
import csv
import os
from datetime import datetime
from scipy.linalg import solve_banded
import sys

# Results file with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("results/semi-implicit-upwind", exist_ok=True)

scheme_name = "semi_implicit_upwind_time_EOC"
results_filename = f"results/semi-implicit-upwind/{scheme_name}_{timestamp}.csv"

print("Saving results to:", os.path.abspath(results_filename))

# CSV saver
def save_run_result(filename, data):
    file_exists = os.path.isfile(filename)

    fieldnames = [
        "scheme",
        "Nt",
        "Nx", "Nv",
        "dx", "dv", "dt",
        "V_large_enough",
        "CFL",
        "positivity",
        "mass_cons",
        "err_L1",
        "err_Linf",
        "approx_L1",
        "approx_Linf",
        "runtime_sec",
        "final_mass",
        "final_min",
        "status",
        "error_message"
    ]

    with open(filename, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter=';')

        if not file_exists:
            writer.writeheader()

        writer.writerow(data)

# Start timer
start_time = time.time()

# Parameters
# Initial condition
x10 = 7
v10 = 15
sigma = 1

# Reset
u_F = 9
u_R = 4
sigma_rho = 0.001

# Grid
x_min = -9
x_max = u_F
size_x = x_max - x_min
X = max(u_F, abs(x_min))

v_min = -25
v_max = 25
size_v = v_max - v_min
V = max(abs(v_min), abs(v_max))

# n = 63
n=2

# Fixed space grid for time-EOC
number_points_space = size_x + 1 + n * size_x
number_points_velocity = size_v + 1 + n * size_v

Nx = number_points_space - 2
Nv = number_points_velocity - 2

delta_x = np.float64(size_x / (Nx + 1))
delta_v = np.float64(size_v / (Nv + 1))

x = np.linspace(x_min, x_max, Nx + 2, dtype=np.float64)
v = np.linspace(v_min, v_max, Nv + 2, dtype=np.float64)

# electric parameters
b = -10
nu = 0.1
omega_0 = 1
tau = 0.6

a_0 = 1
a_1 = 1

T = 1.0

# Time grids for time-EOC
seq_Nt = [1600, 3200, 6400, 12800, 25600, 51200, 102400]
# seq_Nt = [100, 200, 400, 800]

mass_tol = 1e-12

# Completion sounds

USE_SOUND = True


def play_sequence(notes, volume=0.12):
    if not USE_SOUND:
        return

    import tempfile
    import wave
    import shutil
    import subprocess

    fs = 44100
    pieces = []

    # Build ONE continuous waveform (notes + silences)
    for freq, duration in notes:
        n = int(fs * duration)

        if freq is None or freq == 0:
            s = np.zeros(n)
        else:
            t = np.linspace(0.0, duration, n, endpoint=False)
            s = volume * np.sin(2.0 * np.pi * freq * t)

        pieces.append(s)

    signal = np.concatenate(pieces)

    audio = np.int16(signal * 32767)
    stereo = np.column_stack([audio, audio]).ravel()

    path = None

    try:
        # Write temp wav
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            path = tmp.name

        with wave.open(path, "wb") as w:
            w.setnchannels(2)
            w.setsampwidth(2)
            w.setframerate(fs)
            w.writeframes(stereo.tobytes())

        # Platform playback
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME)

        elif sys.platform == "darwin":
            player = shutil.which("afplay")
            if player is None:
                raise RuntimeError("afplay not found")
            subprocess.run([player, path],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        elif sys.platform.startswith("linux"):
            player = shutil.which("paplay") or shutil.which("pw-play")
            if player is None:
                raise RuntimeError("Neither paplay nor pw-play was found")
            subprocess.run([player, path],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

        else:
            raise RuntimeError(f"Unsupported platform for sound: {sys.platform}")

    except Exception as e:
        print(f"\nSound notification failed: {e}")

    finally:
        if path is not None:
            try:
                # Windows may still hold the file briefly
                if sys.platform.startswith("win"):
                    time.sleep(0.2)
                os.remove(path)
            except OSError:
                pass


def play_success_sound():
    play_sequence([
        (700, 0.2),
        (0,   0.1),
        (700, 0.1),
        (0,   0.1),
        (900, 0.8),
    ])


def play_failure_sound():
    play_sequence([
        (300, 0.2),
        (0,   0.1),
        (270, 0.2),
        (0,   0.1),
        (240, 0.2),
        (0,   0.1),
        (225, 0.8),
    ])


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

# Main
try:

    # utilities
    def pos_part(a):
        return abs(a) * (np.sign(a) + 1.0) / 2.0

    def neg_part(a):
        return a * (np.sign(a) - 1.0) / 2.0

    def L1diff(p1, p2, dx=1.0, dv=1.0):
        return dx * dv * np.sum(np.abs(p1 - p2), dtype=np.float64)

    def Linfdiff(p1, p2):
        return np.max(np.abs(p1 - p2))

    def restrict_injection(p_fine):
        return p_fine[::2, ::2].copy()

    def restrict_full_weighting(p_fine):
        nxf, nvf = p_fine.shape

        if ((nxf - 1) % 2 != 0) or ((nvf - 1) % 2 != 0):
            raise ValueError("Fine grid is not compatible with a factor-2 nodal restriction.")

        p_coarse = p_fine[::2, ::2].copy()

        if p_coarse.shape[0] <= 2 or p_coarse.shape[1] <= 2:
            return p_coarse

        p_coarse[1:-1, 1:-1] = (
            4.0 * p_fine[2:-1:2, 2:-1:2]
            + 2.0 * (
                p_fine[1:-2:2, 2:-1:2] +
                p_fine[3::2,   2:-1:2] +
                p_fine[2:-1:2, 1:-2:2] +
                p_fine[2:-1:2, 3::2]
            )
            + (
                p_fine[1:-2:2, 1:-2:2] +
                p_fine[1:-2:2, 3::2]   +
                p_fine[3::2,   1:-2:2] +
                p_fine[3::2,   3::2]
            )
        ) / 16.0

        return p_coarse

    def restrict_fine_to_coarse(p_fine, mode="injection"):
        if mode == "injection":
            return restrict_injection(p_fine)
        elif mode == "full_weighting":
            return restrict_full_weighting(p_fine)
        else:
            raise ValueError(f"Unknown restriction mode: {mode}")

    Activity = []

    if (1.0 / tau**2 - 4.0 * omega_0**2 < 0.0):
        print("We are in the oscillatory framework")
    else:
        raise RuntimeError("We are not in the oscillatory framework, please change the parameters")

    np.set_printoptions(precision=25)

    # helpers & indexing
    def index_x(point):
        return int((point - x_min) / delta_x)

    def index_v(point):
        return int((point - v_min) / delta_v)

    i_R = index_x(u_R)
    i_F = index_x(u_F)
    i_max = index_x(x_max)

    j_0 = index_v(0)
    j_max = index_v(v_max)

    # fixed precomputed arrays
    x_col = x[:, None]
    v_row = v[None, :]

    x_interior_col = x[1:i_F, None]   # rows where A is solved
    v_full_row = v[None, :]
    J_full = np.arange(Nv + 2, dtype=np.int64)[None, :]

    j_neg_x = slice(0, j_0)            # v < 0
    j_pos_x = slice(j_0 + 1, Nv + 2)   # v > 0

    # globals depending on dt
    delta_t = None
    dt_over_dx = None
    dt_over_dv = None
    ab_row = None

    # initial & source
    # initial Gaussian normalized to 1
    inv2s2_init = 1.0 / (2.0 * sigma**2)
    gx_init = np.exp(-(x - x10)**2 * inv2s2_init)
    gv_init = np.exp(-(v - v10)**2 * inv2s2_init)
    p0 = np.multiply.outer(gx_init, gv_init)

    # zero inflow
    p0[0, j_0:] = 0.0
    p0[i_F, :j_0 + 1] = 0.0

    p_initial = p0 / (delta_x * delta_v * np.sum(p0, dtype=np.float64))
    p_initial *= 1.0 / (delta_x * delta_v * np.sum(p_initial, dtype=np.float64))

    print("The initial mass is equal to:", delta_x * delta_v * np.sum(p_initial, dtype=np.float64))

    # source Maxwellian normalized to 1
    inv2s2_src = 1.0 / (2.0 * sigma_rho**2)
    gx_src = np.exp(-(x - u_R)**2 * inv2s2_src)
    gv_src = np.exp(-(v - 0.0)**2 * inv2s2_src)
    rho0 = np.multiply.outer(gx_src, gv_src)

    rho0[0, j_0:] = 0.0
    rho0[i_F, :j_0 + 1] = 0.0

    rho = rho0 / (delta_x * delta_v * np.sum(rho0, dtype=np.float64))
    mass_num_delta = delta_x * delta_v * np.sum(rho, dtype=np.float64)

    # critical index
    def critical_index_array(N):
        jc = ((tau * (-omega_0**2 * x[1:i_F] + b * (nu + N)) + V) / delta_v).astype(np.int64)
        return np.clip(jc, -1, Nv + 1)

    def set_time_step(Nt):
        global delta_t, dt_over_dx, dt_over_dv, ab_row

        delta_t = np.float64(T / (Nt - 1))
        dt_over_dx = delta_t / delta_x
        dt_over_dv = delta_t / delta_v

        ab_row = np.zeros((3, Nv + 2), dtype=np.float64)

    # implicit diffusion matrix in banded row form 
    def build_ab_row(N):
        """
        Solve row-wise in v:
            (I - alpha D_vv) p^{n+1} = RHS
        with Robin-type one-sided diffusion rows at j=0 and j=Nv+1.
        """
        alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

        ab_row.fill(0.0)

        ab_row[1, :] = 1.0 + 2.0 * alpha
        ab_row[0, 1:] = -alpha
        ab_row[2, :-1] = -alpha

        ab_row[1, 0] = 1.0 + alpha
        ab_row[1, -1] = 1.0 + alpha

        return ab_row

    # explicit transport operator
    def apply_B_2d(p, N):
        """
        Explicit transport at time n:
        - x-upwind transport,
        - v-upwind transport,
        - special critical strip j_c, j_c+1,
        - same Robin-type transport boundary treatment.
        """
        out = p.copy()

        # Horizontal transport in x

        # v > 0 : backward difference in x, rows i=1,...,i_F
        if j_0 + 1 < Nv + 2:
            coeff_pos = (-v[j_pos_x] * dt_over_dx)[None, :]
            out[1:i_F + 1, j_pos_x] += coeff_pos * (
                p[1:i_F + 1, j_pos_x] - p[0:i_F, j_pos_x]
            )

        # v < 0 : forward difference in x, rows i=0,...,i_F-1
        if j_0 > 0:
            coeff_neg = (-v[j_neg_x] * dt_over_dx)[None, :]
            out[0:i_F, j_neg_x] += coeff_neg * (
                p[1:i_F + 1, j_neg_x] - p[0:i_F, j_neg_x]
            )

        # Vertical transport in v on interior x-rows only
        if i_F > 1:
            p_int = p[1:i_F, :]
            out_int = out[1:i_F, :]

            muv = - (omega_0**2) * x_interior_col - v_full_row / tau + b * (nu + N)
            beta_v = -muv * dt_over_dv

            # attenuation
            out_int += (delta_t / tau) * p_int

            # critical strip
            jc = critical_index_array(N)
            crit_mask = (J_full == jc[:, None]) | (J_full == (jc[:, None] + 1))

            # remove regular behavior on critical strip so it becomes one-sided
            out_int += np.where(
                crit_mask & (muv > 0.0),
                -(beta_v + delta_t / tau) * p_int,
                0.0
            )
            out_int += np.where(
                crit_mask & (muv < 0.0),
                (beta_v - delta_t / tau) * p_int,
                0.0
            )

            if Nv >= 1:
                # standard upwind on j=1,...,Nv
                out_int[:, 1:Nv+1] += np.where(
                    muv[:, 1:Nv+1] > 0.0,
                    beta_v[:, 1:Nv+1] * (p_int[:, 1:Nv+1] - p_int[:, 0:Nv]),
                    0.0
                )

                out_int[:, 1:Nv+1] += np.where(
                    muv[:, 1:Nv+1] < 0.0,
                    beta_v[:, 1:Nv+1] * (p_int[:, 2:Nv+2] - p_int[:, 1:Nv+1]),
                    0.0
                )

            # Robin-type boundary rows j=0 and j=Nv+1
            mu_bottom = - (omega_0**2) * x[1:i_F] - v[1] / tau + b * (nu + N)
            mu_top = - (omega_0**2) * x[1:i_F] - v[Nv] / tau + b * (nu + N)

            out[1:i_F, 0] += -pos_part(mu_bottom) * dt_over_dv * p[1:i_F, 0]
            out[1:i_F, Nv+1] += -neg_part(mu_top) * dt_over_dv * p[1:i_F, Nv+1]

        return out

    # implicit row-wise solve
    def solve_A_rowwise(D, N):
        """
        Solve A p^{n+1} = D row by row in v.
        Boundary x-rows i=0 and i=i_F are identity rows.
        """
        pnew = np.empty_like(D)

        pnew[0, :] = D[0, :]
        pnew[i_F, :] = D[i_F, :]

        if i_F > 1:
            ab = build_ab_row(N)
            rhs = D[1:i_F, :].T
            sol = solve_banded((1, 1), ab, rhs)
            pnew[1:i_F, :] = sol.T

        return pnew

    def calculate(p):
        p_old = p.copy()

        N = np.float64(
            delta_v * np.sum(p_old[i_F, j_0+1:Nv+1] * v[j_0+1:Nv+1], dtype=np.float64)
            - delta_v * np.sum(p_old[0, 1:j_0] * v[1:j_0], dtype=np.float64)
        )
        Activity.append(N)

        D = apply_B_2d(p_old, N) + N * delta_t * rho / mass_num_delta
        pnew = solve_A_rowwise(D, N)

        return pnew, N

    # main
    seq_sol = []
    seq_runtime = []
    seq_flags = []
    seq_lastN = []
    seq_dt = []

    seq_final_mass = []
    seq_final_min = []
    seq_status = []
    seq_error = []

    for Nt in seq_Nt:
        print(f"\n Nt = {Nt}")
        run_start = time.time()

        set_time_step(Nt)
        seq_dt.append(delta_t)

        p = p_initial.copy()

        V_large_enough = 1
        CFL = 1
        mass_cons = 1
        positivity = 1
        last_N = np.nan

        try:
            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                for k in range(Nt):
                    p, N = calculate(p)
                    last_N = N

                    if CFL:
                        CFL = int(
                            delta_t <= 1.0 / (
                                V * (1.0 / delta_x + 1.0 / (tau * delta_v))
                                + X * omega_0**2 / delta_v
                                + abs(b) * (nu + N) / delta_v
                            )
                        )

                    if V_large_enough:
                        V_large_enough = int(
                            V - delta_v > tau * (X * omega_0**2 + abs(b) * (nu + N))
                        )

                    if (k % max(1, Nt // 5) == 0) and (k > 0):
                        print(f"Nt={Nt}, k={k}/{Nt-1}")

            runtime = time.time() - run_start
            final_mass = delta_x * delta_v * np.sum(p, dtype=np.float64)
            final_min = np.min(p)

            mass_cons = int(abs(final_mass - 1.0) < mass_tol)
            positivity = int(final_min >= 0.0)

            print(f"The final mass is: {final_mass}")
            print(f"The final min is: {final_min}")
            print(
                f"Nt={Nt}, last N={last_N:.6e}, "
                f"CFL={bool(CFL)}, V_large_enough={bool(V_large_enough)}, "
                f"positivity={bool(positivity)}, "
                f"mass_cons={bool(mass_cons)}, "
                f"final_mass={final_mass}, "
                f"final_min={final_min}"
            )

            seq_sol.append(p.copy())
            seq_runtime.append(runtime)
            seq_flags.append((
                bool(V_large_enough),
                bool(CFL),
                bool(positivity),
                bool(mass_cons)
            ))
            seq_lastN.append(last_N)

            seq_final_mass.append(final_mass)
            seq_final_min.append(final_min)
            seq_status.append("success")
            seq_error.append("")

        except FloatingPointError as e:
            runtime = time.time() - run_start

            print(f"Nt={Nt} failed early due to floating-point error: {e}")
            print("Skipping to next Nt.")

            seq_sol.append(None)
            seq_runtime.append(runtime)
            seq_flags.append((False, False, False, False))
            seq_lastN.append(np.nan)

            seq_final_mass.append(np.nan)
            seq_final_min.append(np.nan)
            seq_status.append("failed")
            seq_error.append(str(e))

        except Exception as e:
            runtime = time.time() - run_start

            print(f"Nt={Nt} failed early due to unexpected error: {e}")
            print("Skipping to next Nt.")

            seq_sol.append(None)
            seq_runtime.append(runtime)
            seq_flags.append((False, False, False, False))
            seq_lastN.append(np.nan)

            seq_final_mass.append(np.nan)
            seq_final_min.append(np.nan)
            seq_status.append("failed")
            seq_error.append(f"{type(e).__name__}: {e}")

    errors_L1 = [np.nan] * len(seq_Nt)
    errors_Linf = [np.nan] * len(seq_Nt)

    for i in range(1, len(seq_Nt)):
        p_coarse_t = seq_sol[i - 1]
        p_fine_t = seq_sol[i]

        if (p_coarse_t is not None) and (p_fine_t is not None):
            err_L1 = L1diff(p_fine_t, p_coarse_t, dx=delta_x, dv=delta_v)
            err_Linf = Linfdiff(p_fine_t, p_coarse_t)

            errors_L1[i] = err_L1
            errors_Linf[i] = err_Linf

    # Experimental orders in time
    approx_L1_all = [np.nan] * len(seq_Nt)
    approx_Linf_all = [np.nan] * len(seq_Nt)

    for i in range(2, len(seq_Nt)):
        if (
            np.isfinite(errors_L1[i - 1]) and np.isfinite(errors_L1[i]) and
            errors_L1[i - 1] > 0.0 and errors_L1[i] > 0.0
        ):
            approx_L1_all[i] = np.log2(errors_L1[i - 1] / errors_L1[i])

        if (
            np.isfinite(errors_Linf[i - 1]) and np.isfinite(errors_Linf[i]) and
            errors_Linf[i - 1] > 0.0 and errors_Linf[i] > 0.0
        ):
            approx_Linf_all[i] = np.log2(errors_Linf[i - 1] / errors_Linf[i])

    # Save all runs to CSV
    for idx, Nt in enumerate(seq_Nt):
        V_large_enough_i, CFL_i, positivity_i, mass_cons_i = seq_flags[idx]

        save_run_result(results_filename, {
            "scheme": scheme_name,
            "Nt": Nt,
            "Nx": Nx,
            "Nv": Nv,
            "dx": delta_x,
            "dv": delta_v,
            "dt": seq_dt[idx],
            "V_large_enough": V_large_enough_i,
            "CFL": CFL_i,
            "positivity": positivity_i,
            "mass_cons": mass_cons_i,
            "err_L1": errors_L1[idx],
            "err_Linf": errors_Linf[idx],
            "approx_L1": approx_L1_all[idx],
            "approx_Linf": approx_Linf_all[idx],
            "runtime_sec": seq_runtime[idx],
            "final_mass": seq_final_mass[idx],
            "final_min": seq_final_min[idx],
            "status": seq_status[idx],
            "error_message": seq_error[idx]
        })

    print("The list of pairwise experimental orders of convergence in norm L1 is:")
    for val in approx_L1_all[2:]:
        print(val)

    print("The list of pairwise experimental orders of convergence in norm Linfty is:")
    for val in approx_Linf_all[2:]:
        print(val)

    print("Done. Results saved to CSV.")

    # End timer
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

except Exception as e:
    print("Simulation failed:", e)
    safe_play_failure_sound()
    raise

else:
    print("Simulation completed successfully")
    safe_play_success_sound()