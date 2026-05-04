# -*- coding: utf-8 -*-
"""
Fast explicit upwind version:
- diffusion in v explicit at time n,
- transport explicit at time n,
- upwind finite differences,
- NO critical-strip correction,
- Robin-type treatment at j=0 and j=Nv+1,
- experimental order of convergence in time (fixed space grid),
- CSV saving of results.

- if a floating-point overflow / invalid / divide-by-zero happens
  during one run, that run is aborted immediately,
- the code then continues with the next Nt,
- failed runs are stored as NaN in the CSV.
"""

import numpy as np
import time
import csv
import os
from datetime import datetime
import sys
import subprocess

# ============================================================
# Results file with timestamp
# ============================================================
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs("Results/Explicit_upwind", exist_ok=True)

scheme_name = "Current_fast_explicit_upwind_time_EOC_no_jc"
results_filename = f"Results/Explicit_upwind/{scheme_name}_{timestamp}.csv"

print("Saving results to:", os.path.abspath(results_filename))

# ---------------- CSV saver ----------------
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

# ---------------- Start timer ----------------
start_time = time.time()

# ============================================================
# Parameters
# ============================================================

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

# n = 3
n = 63

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
# seq_Nt = [100, 200, 400, 800]
# seq_Nt = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200]
# seq_Nt = [800, 1600, 3200, 6400, 12800]
seq_Nt = [1600, 3200, 6400, 12800, 25600, 51200, 102400]

mass_tol = 1e-12

# ============================================================
# Completion sound
# ============================================================
def play_sound(filename):
    if sys.platform.startswith("win"):
        os.startfile(filename)
    elif sys.platform == "darwin":
        subprocess.run(["afplay", filename])
    else:
        subprocess.run(["xdg-open", filename])

base_dir = os.path.dirname(os.path.abspath(__file__))

sound_success = os.path.normpath(os.path.join(
    base_dir, "..", "..", "Sound files", "superman.mp3"
))

sound_failure = os.path.normpath(os.path.join(
    base_dir, "..", "..", "Sound files", "duel_of_the_fates.mp3"
))

# ============================================================
# Main execution
# ============================================================
try:
    # ---------------------- utilities ----------------------
    def pos_part(a):
        return abs(a) * (np.sign(a) + 1.0) / 2.0

    def neg_part(a):
        return a * (np.sign(a) - 1.0) / 2.0

    def L1diff(p1, p2, dx=1.0, dv=1.0):
        return dx * dv * np.sum(np.abs(p1 - p2), dtype=np.float64)

    def Linfdiff(p1, p2):
        return np.max(np.abs(p1 - p2))

    if (1.0 / tau**2 - 4.0 * omega_0**2 < 0.0):
        print("We are in the oscillatory framework")
    else:
        raise RuntimeError("We are not in the oscillatory framework, please change the parameters")

    np.set_printoptions(precision=25)

    # ---------------------- helpers & indexing ----------------------
    def index_x(point):
        return int((point - x_min) / delta_x)

    def index_v(point):
        return int((point - v_min) / delta_v)

    i_R = index_x(u_R)
    i_F = index_x(u_F)
    i_max = index_x(x_max)

    j_0 = index_v(0)
    j_max = index_v(v_max)

    # ---------------------- fixed precomputed arrays ----------------------
    x_col = x[:, None]
    v_row = v[None, :]

    x_interior_col = x[1:i_F, None]
    v_full_row = v[None, :]

    j_neg_x = slice(0, j_0)            # v < 0
    j_pos_x = slice(j_0 + 1, Nv + 2)   # v > 0

    # ---------------------- globals depending on dt ----------------------
    delta_t = None
    dt_over_dx = None
    dt_over_dv = None

    # ---------------------- initial & source ----------------------
    inv2s2_init = 1.0 / (2.0 * sigma**2)
    gx_init = np.exp(-(x - x10)**2 * inv2s2_init)
    gv_init = np.exp(-(v - v10)**2 * inv2s2_init)
    p0 = np.multiply.outer(gx_init, gv_init)

    # inflow-zeroing
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

    # ---------------------- set dt-dependent quantities ----------------------
    def set_time_step(Nt):
        global delta_t, dt_over_dx, dt_over_dv

        delta_t = np.float64(T / (Nt - 1))
        dt_over_dx = delta_t / delta_x
        dt_over_dv = delta_t / delta_v

    # ---------------- explicit full operator ----------------
    def apply_scheme_2d(p, N):
        """
        Fully explicit operator:
        - explicit diffusion in v,
        - x-upwind transport,
        - v-upwind transport,
        - NO critical-strip correction,
        - Robin-type treatment at j=0 and j=Nv+1.
        """
        out = p.copy()

        # ============================================================
        # 0) Explicit diffusion in v
        # ============================================================
        if i_F > 1:
            alpha = (a_0 + a_1 * N) * delta_t / (delta_v**2)

            out[1:i_F, 1:Nv+1] += alpha * (
                p[1:i_F, 0:Nv] - 2.0 * p[1:i_F, 1:Nv+1] + p[1:i_F, 2:Nv+2]
            )

            out[1:i_F, 0] += alpha * (p[1:i_F, 1] - p[1:i_F, 0])
            out[1:i_F, Nv+1] += alpha * (p[1:i_F, Nv] - p[1:i_F, Nv+1])

        # ============================================================
        # 1) Horizontal transport in x
        # ============================================================
        if j_0 + 1 < Nv + 2:
            coeff_pos = (-v[j_pos_x] * dt_over_dx)[None, :]
            out[1:i_F + 1, j_pos_x] += coeff_pos * (
                p[1:i_F + 1, j_pos_x] - p[0:i_F, j_pos_x]
            )

        if j_0 > 0:
            coeff_neg = (-v[j_neg_x] * dt_over_dx)[None, :]
            out[0:i_F, j_neg_x] += coeff_neg * (
                p[1:i_F + 1, j_neg_x] - p[0:i_F, j_neg_x]
            )

        # ============================================================
        # 2) Vertical transport in v on interior x-rows only
        # ============================================================
        if i_F > 1:
            p_int = p[1:i_F, :]
            out_int = out[1:i_F, :]

            muv = - (omega_0**2) * x_interior_col - v_full_row / tau + b * (nu + N)
            beta_v = -muv * dt_over_dv

            out_int += (delta_t / tau) * p_int

            if Nv >= 1:
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

            mu_bottom = - (omega_0**2) * x[1:i_F] - v[1] / tau + b * (nu + N)
            mu_top = - (omega_0**2) * x[1:i_F] - v[Nv] / tau + b * (nu + N)

            out[1:i_F, 0] += -pos_part(mu_bottom) * dt_over_dv * p[1:i_F, 0]
            out[1:i_F, Nv+1] += -neg_part(mu_top) * dt_over_dv * p[1:i_F, Nv+1]

        return out

    # ---------------- one step ----------------
    def calculate(p):
        p_old = p.copy()

        N = np.float64(
            delta_v * np.sum(p_old[i_F, j_0+1:Nv+1] * v[j_0+1:Nv+1], dtype=np.float64)
            - delta_v * np.sum(p_old[0, 1:j_0] * v[1:j_0], dtype=np.float64)
        )

        pnew = apply_scheme_2d(p_old, N) + N * delta_t * rho / mass_num_delta
        return pnew, N

    # ================= MAIN LOOP =================
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
        print(f"\n===== Nt = {Nt} =====")
        run_start = time.time()

        set_time_step(Nt)
        seq_dt.append(delta_t)

        p = p_initial.copy()

        V_large_enough = 1
        CFL = 1
        positivity = 1
        mass_cons = 1
        last_N = np.nan

        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
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
                f"CFL={bool(CFL)}, "
                f"V_large_enough={bool(V_large_enough)}, "
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

    # ------------------------------------------------------------
    # Pairwise self-convergence in time
    # ------------------------------------------------------------
    errors_L1 = [np.nan] * len(seq_Nt)
    errors_Linf = [np.nan] * len(seq_Nt)

    for i in range(1, len(seq_Nt)):
        p_coarse_t = seq_sol[i - 1]
        p_fine_t = seq_sol[i]

        if (p_coarse_t is not None) and (p_fine_t is not None):
            errors_L1[i] = L1diff(p_fine_t, p_coarse_t, dx=delta_x, dv=delta_v)
            errors_Linf[i] = Linfdiff(p_fine_t, p_coarse_t)

    # ------------------------------------------------------------
    # Experimental orders in time
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # Save all runs to CSV
    # ------------------------------------------------------------
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

    # ---------------- End timer ----------------
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

except Exception as e:
    print("Simulation failed:", e)
    play_sound(sound_failure)
    raise

else:
    print("Simulation completed successfully")
    play_sound(sound_success)