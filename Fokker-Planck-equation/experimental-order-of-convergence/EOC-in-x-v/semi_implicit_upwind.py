# -*- coding: utf-8 -*-
"""
Semi-implicit upwind:
- critical strip j_c, j_c+1 treatment,
- CSV saving of results,
- experimental order computed by pairwise self-convergence:
    compare h/2 to h on the h-grid,
    compare h/4 to h/2 on the h/2-grid, etc.

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
from scipy.linalg import solve_banded
import time
import csv
import os
from datetime import datetime
import sys
import subprocess

# User choice for restriction in the convergence study
restriction_mode = "injection"
# restriction_mode = "full_weighting"

# results file with timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("results/semi-implicit-upwind", exist_ok=True)

scheme_name = f"semi_implicit_upwind_EOC_xv_{restriction_mode}"
results_filename = f"results/semi-implicit-upwind/{scheme_name}_{timestamp}.csv"

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

if (1 / tau**2 - 4 * omega_0**2 < 0):
    print("We are in the oscillatory framework")
else:
    raise RuntimeError("We are not in the oscillatory framework, please change the parameters")

# T = 0.0001
T = 1

# Nt = 1001
Nt = 100001

delta_t = np.float64(T / (Nt - 1))

mass_tol = 1e-12

# Completion sound
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
        import tempfile
        import wave

        fs = 44100
        n_samples = int(fs * duration)
        t = np.linspace(0.0, duration, n_samples, endpoint=False)

        signal = volume * np.sin(2.0 * np.pi * frequency * t)

        audio_mono = np.int16(signal * 32767)
        audio_stereo = np.column_stack([audio_mono, audio_mono]).ravel()

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
        

# Main
try:

    # utilities
    def L1diff(p1, p2, dx=1.0, dv=1.0):
        return dx * dv * np.sum(np.abs(p1 - p2), dtype=np.float64)

    def Linfdiff(p1, p2):
        return np.max(np.abs(p1 - p2))

    def restrict_injection(p_fine):
        """
        Nodal injection from fine grid to coarse grid:
        coarse node (i,j) <- fine node (2i,2j)
        """
        return p_fine[::2, ::2].copy()

    def restrict_full_weighting(p_fine):
        """
        2D nodal full-weighting restriction from fine grid to coarse grid.
        Boundary coarse nodes are obtained by injection.
        Interior coarse nodes use the 3x3 stencil:
            1 2 1
            2 4 2   / 16
            1 2 1
        centered at the coinciding fine node.
        """
        nxf, nvf = p_fine.shape

        if ((nxf - 1) % 2 != 0) or ((nvf - 1) % 2 != 0):
            raise ValueError("Fine grid is not compatible with a factor-2 nodal restriction.")

        p_coarse = p_fine[::2, ::2].copy()

        # If there are no interior coarse nodes, injection is all we can do
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

    ab_row = None
    J_full = None
    x_interior_col = None
    v_interior_row = None

    # init
    def init_xv(n):
        global Nx, Nv, delta_x, delta_v, dt_over_dx, dt_over_dv
        global x, v, x_col, v_row
        global p_initial, rho, i_F, j_0
        global j_pos_x, j_neg_x, interior_rows
        global ab_row, J_full, x_interior_col, v_interior_row

        # Same grid logic as old code
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

        # row-wise implicit matrix A
        ab_row = np.zeros((3, Nv + 2), dtype=np.float64)

        # j-array reused in critical-strip construction
        J_full = np.arange(Nv + 2, dtype=np.int64)[None, :]

    def build_ab_row(N):
        """
        Row-wise tridiagonal matrix in v:
        - interior rows i = 1,...,i_F-1
        - main diagonal 1+2alpha
        - one-sided at j=0 and j=Nv+1: 1+alpha
        """
        alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

        ab_row.fill(0.0)

        ab_row[1, :] = 1.0 + 2.0 * alpha

        # solve_banded convention
        ab_row[0, 1:] = -alpha
        ab_row[2, :-1] = -alpha

        # v-boundaries
        ab_row[1, 0] = 1.0 + alpha
        ab_row[1, -1] = 1.0 + alpha

        return ab_row

    # transport operator B
    def apply_B_2d(p, N):
        out = p.copy()

        # Horizontal transport in x
        # v > 0 : backward difference in x, valid for rows i=1,...,i_F
        if j_0 + 1 < Nv + 2:
            coeff_pos = (-v[j_pos_x] * dt_over_dx)[None, :]
            out[1:i_F + 1, j_pos_x] += coeff_pos * (
                p[1:i_F + 1, j_pos_x] - p[0:i_F, j_pos_x]
            )

        # v < 0 : forward difference in x, valid for rows i=0,...,i_F-1
        if j_0 > 0:
            coeff_neg = (-v[j_neg_x] * dt_over_dx)[None, :]
            out[0:i_F, j_neg_x] += coeff_neg * (
                p[1:i_F + 1, j_neg_x] - p[0:i_F, j_neg_x]
            )

        # Vertical transport in v (only on interior x-rows)
        if i_F > 1:
            p_int = p[1:i_F, :]
            out_int = out[1:i_F, :]

            muv = - (omega_0**2) * x_interior_col - v_interior_row / tau + b * (nu + N)
            beta_v = -muv * dt_over_dv

            # attenuation
            out_int += (delta_t / tau) * p_int

            # critical strip
            jc = ((tau * (-omega_0**2 * x[1:i_F] + b * (nu + N)) + V) / delta_v).astype(np.int64)
            jc = np.clip(jc, -1, Nv + 1)

            crit_mask = (J_full == jc[:, None]) | (J_full == (jc[:, None] + 1))

            # corrections on critical strip
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

            # standard upwind in v on j=1,...,Nv
            if Nv >= 1:
                out_int[:, 1:Nv + 1] += np.where(
                    muv[:, 1:Nv + 1] > 0.0,
                    beta_v[:, 1:Nv + 1] * (p_int[:, 1:Nv + 1] - p_int[:, 0:Nv]),
                    0.0
                )

                out_int[:, 1:Nv + 1] += np.where(
                    muv[:, 1:Nv + 1] < 0.0,
                    beta_v[:, 1:Nv + 1] * (p_int[:, 2:Nv + 2] - p_int[:, 1:Nv + 1]),
                    0.0
                )

            # v-boundaries j=0 and j=Nv+1
            mu_bottom = - (omega_0**2) * x[1:i_F] - v[0] / tau + b * (nu + N)
            mu_top = - (omega_0**2) * x[1:i_F] - v[Nv + 1] / tau + b * (nu + N)

            out[1:i_F, 0] += -mu_bottom * dt_over_dv * p[1:i_F, 0]
            out[1:i_F, Nv + 1] += mu_top * dt_over_dv * p[1:i_F, Nv + 1]

        return out

    # implicit row-wise solve
    def solve_A_rowwise(D2, N):
        """
        Solve A p^{n+1} = D2 row by row in v.
        Boundary x-rows i=0 and i=i_F remain identity rows.
        """
        pnew = np.empty_like(D2)

        pnew[0, :] = D2[0, :]
        pnew[i_F, :] = D2[i_F, :]

        if i_F > 1:
            ab = build_ab_row(N)
            rhs = D2[1:i_F, :].T
            sol = solve_banded((1, 1), ab, rhs)
            pnew[1:i_F, :] = sol.T

        return pnew

    def calculate(p, k):
        if k == 0:
            p = p_initial.copy()

        # same activity formula as old code
        N = np.float64(
            delta_v * np.sum(p[i_F, j_0 + 1:Nv + 2] * v[j_0 + 1:Nv + 2], dtype=np.float64)
            - delta_v * np.sum(p[0, 0:j_0] * v[0:j_0], dtype=np.float64)
        )

        Activity.append(N)

        D2 = apply_B_2d(p, N) + N * delta_t * rho
        pnew = solve_A_rowwise(D2, N)

        return pnew, N

    # Main

    # seq_Nxv = [0, 1, 3, 7]
    seq_Nxv = [0, 1, 3, 7, 15, 31, 63, 127, 255,511]
    
    # storage of final solutions only
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

        try:
            init_xv(n)
            p = p_initial.copy()

            CFL = 1
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

                    # if (k % 50000 == 0) and (k > 0):
                    #     print(f"n={n}, k={k}/{Nt-1}")

            runtime = time.time() - run_start
            final_mass = delta_x * delta_v * np.sum(p, dtype=np.float64)
            final_min = np.min(p)

            print(f"The final mass is: {final_mass}")
            print(f"The final min is: {final_min}")

            print(
                f"n={n}, Nx={Nx}, Nv={Nv}, last N={last_N:.6e}, "
                f"CFL={bool(CFL)}, V_large_enough={bool(V_large_enough)}, "
                f"positivity={bool(positivity)}, mass_cons={bool(mass_cons)}, "
                f"final_mass={final_mass}, final_min={final_min}"
            )

            seq_final_mass.append(final_mass)
            seq_final_min.append(final_min)

            seq_sol.append((p.copy(), x.copy(), v.copy()))
            seq_runtime.append(runtime)
            seq_flags.append((bool(V_large_enough), bool(CFL), bool(positivity), bool(mass_cons)))
            seq_lastN.append(last_N)
            seq_grid_info.append((Nx, Nv, delta_x, delta_v))
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
            seq_grid_info.append((Nx, Nv, delta_x, delta_v))
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
            seq_grid_info.append((Nx, Nv, delta_x, delta_v))
            seq_status.append("failed")
            seq_error.append(f"{type(e).__name__}: {e}")

    # Pairwise self-convergence:
    # compare level i (fine) to level i-1 (coarse),
    # after restricting fine solution onto the coarse grid.

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

            errors_L1[i] = L1diff(p_fine_on_coarse, p_coarse, dx=dx_coarse, dv=dv_coarse)
            errors_Linf[i] = Linfdiff(p_fine_on_coarse, p_coarse)

    # Experimental orders:
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

    # Save all runs to CSV now that errors/orders are known
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