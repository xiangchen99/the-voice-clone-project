#!/usr/bin/env python3
"""Launcher for the Digital Self Web UI Dashboard.

Zero external pip dependencies required to launch the UI: runs on standard library http.server.
When .venv is available with neural packages, automatically re-executes inside .venv.
Usage:
    python3 run_ui.py [--port 8080] [--host 127.0.0.1]
"""

import argparse
import os
import sys
from pathlib import Path


def check_and_switch_venv() -> None:
    """If a dedicated .venv exists and we are not in it, switch to it."""
    venv_python = Path(__file__).resolve().parent / ".venv" / "bin" / "python"
    if venv_python.exists():
        try:
            curr_exe = Path(sys.executable).resolve()
            target_exe = venv_python.resolve()
            if curr_exe != target_exe:
                print(f"[Launcher] Detected isolated environment at {venv_python}")
                print(f"[Launcher] Re-executing with {target_exe}...")
                os.execv(str(target_exe), [str(target_exe)] + sys.argv)
        except Exception:
            pass


def print_startup_banner(host: str, port: int) -> None:
    """Prints diagnostic telemetry and environment status on server boot."""
    print("=" * 60)
    print(" 🎙️  Digital Self Studio: Voice Cloning & Persona Dashboard")
    print("=" * 60)
    print(f" • Python Runtime: {sys.version.split()[0]} ({sys.executable})")

    # Check TTS & GPU status
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_ok else "None (CPU)"
        print(f" • PyTorch / Compute: {'CUDA Active' if cuda_ok else 'CPU Only'} ({gpu_name})")
    except ImportError:
        print(" • PyTorch / Compute: Not installed in active interpreter (running lightweight mode)")

    try:
        from TTS.api import TTS  # type: ignore
        print(" • Voice Cloning Engine: Coqui XTTS-v2 Neural Pipeline [ONLINE]")
    except ImportError:
        print(" • Voice Cloning Engine: Mock Fallback [OFFLINE]")
        print("   -> To activate authentic RTX 4080 neural voice cloning, run:")
        print("      bash scripts/setup_environment.sh")

    print("-" * 60)
    print(f" • Dashboard UI: http://{host}:{port}")
    print("=" * 60)


def main() -> None:
    check_and_switch_venv()

    parser = argparse.ArgumentParser(description="Digital Self Web UI Dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    args = parser.parse_args()

    print_startup_banner(args.host, args.port)

    from web.server import run_server
    run_server(port=args.port, host=args.host)


if __name__ == "__main__":
    main()

