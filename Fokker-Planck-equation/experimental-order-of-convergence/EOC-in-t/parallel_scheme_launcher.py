# -*- coding: utf-8 -*-

import subprocess
import os
import sys
import numpy as np
import time
from pathlib import Path
from datetime import datetime

base_dir = Path(__file__).resolve().parent

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


try:

    scripts = [
        base_dir / "semi_implicit_upwind.py",
        base_dir / "explicit_upwind.py",
        base_dir / "explicit_centered.py",
        base_dir / "semi_implicit_centered.py",
    ]

    env = os.environ.copy()

    # avoid oversubscription
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # isolate Python from user-site / foreign Python installs
    env["PYTHONNOUSERSITE"] = "1"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)

    log_dir = base_dir / "parallel_logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # First validate all scripts.

    missing_scripts = [script_path for script_path in scripts if not script_path.exists()]

    if missing_scripts:
        print("\nMissing script(s):")
        for script_path in missing_scripts:
            print(f"  {script_path}")
        raise FileNotFoundError("At least one script is missing. No job was launched.")

    # Launch all jobs only after validation succeeded.

    procs = []

    for script_path in scripts:
        log_file = log_dir / f"{script_path.stem}_{timestamp}.log"
        f = open(log_file, "w", encoding="utf-8")

        print(f"Launching: {script_path.name}")

        p = subprocess.Popen(
            [sys.executable, "-I", str(script_path)],
            cwd=str(base_dir),
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
        )

        procs.append((script_path.name, p, f, log_file))

    print("\nAll jobs launched.\n")

    # Wait for all jobs, collect failures.

    failed = []

    for name, p, f, log_file in procs:
        code = p.wait()
        f.close()

        print(f"{name} finished with return code {code}")
        print(f"Log: {log_file}\n")

        if code != 0:
            failed.append((name, code, log_file))

    if failed:
        print("Some jobs failed:")
        for name, code, log_file in failed:
            print(f"  {name}: return code {code}, log: {log_file}")
        raise RuntimeError("At least one child script failed.")

    print("Done.")

except Exception as e:
    print("\nSimulation failed:", e)
    safe_play_failure_sound()
    raise

else:
    print("\nSimulation completed successfully.")
    safe_play_success_sound()