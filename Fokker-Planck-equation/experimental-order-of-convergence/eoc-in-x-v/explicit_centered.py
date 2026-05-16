# -*- coding: utf-8 -*-
"""
Explicit centered:
- centered x-transport in the interior, one-sided at x-boundaries exactly as in the slow code,
- centered v-transport away from the critical strip,
- explicit diffusion in v,
- CSV saving of results,

Restriction choices:
- "injection"      : p_fine_on_coarse = p_fine[::2, ::2]
- "full_weighting" : nodal full-weighting in 2D, with injection on boundaries

- Nt is fixed,
- each n in seq_Nxv is handled independently,
- if a floating-point overflow / invalid / divide-by-zero happens
  during one run, that run is aborted immediately,
- the code then continues with the next n,
- failed runs are stored as nan in the CSV.
"""

import numpy as np
import time
import csv
import os
from datetime import datetime
import sys

# User choice for restriction in the convergence study
restriction_mode = "injection"
# restriction_mode = "full_weighting"

# results file with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("results/explicit-centered", exist_ok=True)

scheme_name = f"explicit_centered_EOC_xv_{restriction_mode}"
results_filename = f"results/explicit-centered/{scheme_name}_{timestamp}.csv"

print("Saving results to:", os.path.abspath(results_filename))

# CSV saver
def save_run_result(filename, data):
    file_exists = os.path.isfile(filename)

    fieldnames = [
        "scheme",
        "n",
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

# Grid box
x_min = -9
x_max = u_F
size_x = x_max - x_min
X = max(u_F, abs(x_min))

v_min = -25
v_max = 25
size_v = v_max - v_min
V = max(abs(v_min), abs(v_max))

# Electric parameters
b = -10
nu = 0.1
omega_0 = 1
tau = 0.6

a_0 = 1
a_1 = 1

if (1.0 / tau**2 - 4.0 * omega_0**2 < 0.0):
    print("We are in the oscillatory framework")
else:
    raise RuntimeError("We are not in the oscillatory framework, please change the parameters")

T = 1
# Nt = 1001
Nt = 100001

delta_t = np.float64(T / (Nt - 1))

mass_tol = 1e-10

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

    np.set_printoptions(precision=25)

    # globals
    Nx = Nv = None
    delta_x = delta_v = None
    dt_over_dx = dt_over_dv = None

    x = v = None
    x_col = v_row = None

    p_initial = rho = None
    i_F = j_0 = None

    j_pos_x = j_neg_x = None
    interior_rows = None

    J_full = None
    x_interior_col = None
    v_interior_row = None

    # init
    def init_xv(n):
        global Nx, Nv, delta_x, delta_v, dt_over_dx, dt_over_dv
        global x, v, x_col, v_row
        global p_initial, rho, i_F, j_0
        global j_pos_x, j_neg_x, interior_rows
        global J_full, x_interior_col, v_interior_row

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

        interior_rows = slice(1, i_F)
        x_interior_col = x[1:i_F, None]
        v_interior_row = v[None, :]

        j_pos_x = slice(j_0 + 1, Nv + 2)   # v > 0
        j_neg_x = slice(0, j_0)            # v < 0

        # initial condition
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

        # source
        inv2s2_src = 1.0 / (2.0 * sigma_rho**2)
        gx_src = np.exp(-(x - u_R)**2 * inv2s2_src)
        gv_src = np.exp(-(v - 0.0)**2 * inv2s2_src)
        rho0 = np.multiply.outer(gx_src, gv_src)

        rho0[0, j_0:] = 0.0
        rho0[i_F, :j_0 + 1] = 0.0

        rho = rho0 / (delta_x * delta_v * np.sum(rho0, dtype=np.float64))

        J_full = np.arange(Nv + 2, dtype=np.int64)[None, :]

    # transport + explicit diffusion operator in 2D
    def apply_B_2d(p, N):
        out = p.copy()
        # Explicit diffusion in v
        if i_F > 1:
            alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

            # interior in v: j = 1,...,Nv
            out[1:i_F, 1:Nv+1] += alpha * (
                p[1:i_F, 0:Nv] - 2.0 * p[1:i_F, 1:Nv+1] + p[1:i_F, 2:Nv+2]
            )

            # bottom boundary j = 0
            out[1:i_F, 0] += alpha * (p[1:i_F, 1] - p[1:i_F, 0])

            # top boundary j = Nv+1
            out[1:i_F, Nv+1] += alpha * (p[1:i_F, Nv] - p[1:i_F, Nv+1])

        # Horizontal transport in x
        # interior rows i = 1,...,Nx : centered in x
        if Nx >= 1:
            coeff_centered_x = (-v[None, :] * dt_over_dx) / 2.0
            out[1:Nx+1, :] += coeff_centered_x * (
                p[2:Nx+2, :] - p[0:Nx, :]
            )

        # left boundary i = 0 : one-sided only if mux < 0
        if j_0 > 0:
            coeff_left = (-v[j_neg_x] * dt_over_dx)[None, :]
            out[0:1, j_neg_x] += coeff_left * (
                p[1:2, j_neg_x] - p[0:1, j_neg_x]
            )

        # right boundary i = Nx+1 : one-sided only if mux > 0
        if j_0 + 1 < Nv + 2:
            coeff_right = (-v[j_pos_x] * dt_over_dx)[None, :]
            out[Nx+1:Nx+2, j_pos_x] += coeff_right * (
                p[Nx+1:Nx+2, j_pos_x] - p[Nx:Nx+1, j_pos_x]
            )

        # Vertical transport in v (only on interior x-rows)
        if i_F > 1:
            p_int = p[1:i_F, :]
            out_int = out[1:i_F, :]

            muv = - (omega_0**2) * x_interior_col - v_interior_row / tau + b * (nu + N)
            beta_v = -muv * dt_over_dv

            # attenuation
            out_int += (delta_t / tau) * p_int

            if Nv >= 1:
                # regular centered transport on j=1,...,Nv
                out_int[:, 1:Nv+1] += 0.5 * beta_v[:, 1:Nv+1] * (
                    p_int[:, 2:Nv+2] - p_int[:, 0:Nv]
                )

            # v-boundaries j=0 and j=Nv+1
            mu_bottom = - (omega_0**2) * x[1:i_F] - v[1] / tau + b * (nu + N)
            mu_top = - (omega_0**2) * x[1:i_F] - v[Nv] / tau + b * (nu + N)

            out[1:i_F, 0] += -pos_part(mu_bottom) * dt_over_dv * p[1:i_F, 0]
            out[1:i_F, Nv+1] += -neg_part(mu_top) * dt_over_dv * p[1:i_F, Nv+1]

        return out

    def calculate(p, k):
        N = np.float64(
            delta_v * np.sum(p[i_F, j_0 + 1:Nv + 2] * v[j_0 + 1:Nv + 2], dtype=np.float64)
            - delta_v * np.sum(p[0, 0:j_0] * v[0:j_0], dtype=np.float64)
        )

        pnew = apply_B_2d(p, N) + N * delta_t * rho

        return pnew, N

    # Main
    # seq_Nxv = [0, 1, 3, 7]
    seq_Nxv = [0, 1, 3, 7, 15, 31, 63, 127, 255, 511]

    seq_sol = []
    seq_runtime = []
    seq_flags = []
    seq_lastN = []
    seq_grid_info = []

    seq_final_mass = []
    seq_final_min = []
    seq_status = []
    seq_error = []

    for n in seq_Nxv:
        print(f"\n n = {n}")
        run_start = time.time()

        Nx_run = np.nan
        Nv_run = np.nan
        dx_run = np.nan
        dv_run = np.nan

        try:
            init_xv(n)
            Nx_run, Nv_run, dx_run, dv_run = Nx, Nv, delta_x, delta_v

            p = p_initial.copy()

            CFL = 1
            CFL_diffusion = 1
            V_large_enough = 1
            mass_cons = 1
            positivity = 1

            last_N = np.nan

            with np.errstate(over='raise', invalid='raise', divide='raise', under='ignore'):
                for k in range(Nt):
                    p, N = calculate(p, k)
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

                    if (k % 1000 == 0) or (k == Nt - 1):
                        if mass_cons:
                            mass_now = delta_x * delta_v * np.sum(p, dtype=np.float64)
                            mass_cons = int(abs(mass_now - 1.0) < mass_tol)

                        if positivity:
                            positivity = int(np.min(p) >= 0.0)

                    if (k % 50000 == 0) and (k > 0):
                        print(f"n={n}, k={k}/{Nt-1}")

            runtime = time.time() - run_start
            final_mass = delta_x * delta_v * np.sum(p, dtype=np.float64)
            final_min = p.min()

            print(f"The final mass is: {final_mass}")
            print(f"The final min is: {final_min}")

            print(
                f"n={n}, Nx={Nx}, Nv={Nv}, last N={last_N:.6e}, "
                f"CFL={bool(CFL)}, "
                f"V_large_enough={bool(V_large_enough)}, "
                f"positivity={bool(positivity)}, "
                f"mass_cons={bool(mass_cons)}, "
                f"final_mass={final_mass}, "
                f"final_min={final_min}"
            )

            seq_final_mass.append(final_mass)
            seq_final_min.append(final_min)

            seq_sol.append((p.copy(), x.copy(), v.copy()))
            seq_runtime.append(runtime)
            seq_flags.append((
                bool(V_large_enough),
                bool(CFL),
                bool(positivity),
                bool(mass_cons)
            ))
            seq_lastN.append(last_N)
            seq_grid_info.append((Nx_run, Nv_run, dx_run, dv_run))
            seq_status.append("success")
            seq_error.append("")

        except FloatingPointError as e:
            runtime = time.time() - run_start

            print(f"n={n} failed early due to floating-point error: {e}")
            print("Skipping to next n.")

            seq_final_mass.append(np.nan)
            seq_final_min.append(np.nan)

            seq_sol.append(None)
            seq_runtime.append(runtime)
            seq_flags.append((False, False, False, False))
            seq_lastN.append(np.nan)
            seq_grid_info.append((Nx_run, Nv_run, dx_run, dv_run))
            seq_status.append("failed")
            seq_error.append(str(e))

        except Exception as e:
            runtime = time.time() - run_start

            print(f"n={n} failed early due to unexpected error: {e}")
            print("Skipping to next n.")

            seq_final_mass.append(np.nan)
            seq_final_min.append(np.nan)

            seq_sol.append(None)
            seq_runtime.append(runtime)
            seq_flags.append((False, False, False, False))
            seq_lastN.append(np.nan)
            seq_grid_info.append((Nx_run, Nv_run, dx_run, dv_run))
            seq_status.append("failed")
            seq_error.append(f"{type(e).__name__}: {e}")

    errors_L1 = [np.nan] * len(seq_Nxv)
    errors_Linf = [np.nan] * len(seq_Nxv)

    for i in range(1, len(seq_Nxv)):
        coarse_entry = seq_sol[i - 1]
        fine_entry = seq_sol[i]

        if (coarse_entry is not None) and (fine_entry is not None):
            p_coarse, x_coarse, v_coarse = coarse_entry
            p_fine, x_fine, v_fine = fine_entry

            dx_coarse = x_coarse[1] - x_coarse[0]
            dv_coarse = v_coarse[1] - v_coarse[0]

            p_fine_on_coarse = restrict_fine_to_coarse(p_fine, mode=restriction_mode)

            if p_fine_on_coarse.shape != p_coarse.shape:
                raise ValueError(
                    f"Restricted fine solution shape {p_fine_on_coarse.shape} "
                    f"does not match coarse solution shape {p_coarse.shape}"
                )

            err_L1 = L1diff(p_fine_on_coarse, p_coarse, dx=dx_coarse, dv=dv_coarse)
            err_Linf = Linfdiff(p_fine_on_coarse, p_coarse)

            errors_L1[i] = err_L1
            errors_Linf[i] = err_Linf

    # Experimental orders
    approx_L1_all = [np.nan] * len(seq_Nxv)
    approx_Linf_all = [np.nan] * len(seq_Nxv)

    for i in range(2, len(seq_Nxv)):
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
    for idx, n in enumerate(seq_Nxv):
        Nx_i, Nv_i, dx_i, dv_i = seq_grid_info[idx]
        V_large_enough_i, CFL_i, positivity_i, mass_cons_i = seq_flags[idx]

        save_run_result(results_filename, {
            "scheme": scheme_name,
            "n": n,
            "Nx": Nx_i,
            "Nv": Nv_i,
            "dx": dx_i,
            "dv": dv_i,
            "dt": delta_t,
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

    print(f"Restriction mode used for EOC: {restriction_mode}")
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