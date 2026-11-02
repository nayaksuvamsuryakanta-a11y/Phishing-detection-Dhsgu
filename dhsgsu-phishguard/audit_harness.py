"""Launch-mode audit harness. Runs each documented launch mode as a
subprocess and prints a clean result table. No shell chaining."""
import subprocess
import sys

VENV = ".venv\\Scripts\\python.exe"
STREAMLIT = ".venv\\Scripts\\streamlit.exe"
ROOT = "."

CASES = [
    ("A", "python data/generate_dataset.py",
     [VENV, "data/generate_dataset.py"], ROOT),
    ("B", "python -m src.train",
     [VENV, "-m", "src.train"], ROOT),
    ("C", "python src/train.py",
     [VENV, "src/train.py"], ROOT),
    ("D", "python -m src.cli --file audit_phish.txt",
     [VENV, "-m", "src.cli", "--file", "audit_phish.txt"], ROOT),
    ("E", "python src/cli.py --file audit_phish.txt",
     [VENV, "src/cli.py", "--file", "audit_phish.txt"], ROOT),
    ("F", "streamlit run app/streamlit_app.py (headless, 20s)",
     [STREAMLIT, "run", "app/streamlit_app.py", "--server.headless=true",
      "--server.port=8599"], ROOT),
]


def run(cmd, cwd, timeout=240):
    try:
        p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                           timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "TIMEOUT"
    except Exception as exc:  # pragma: no cover - harness only
        return 1, "", f"EXCEPTION: {exc}"


def main():
    print(f"{'ID':<3} {'launch mode':<40} {'rc':<4} status")
    print("-" * 72)
    for label, desc, cmd, cwd in CASES:
        rc, out, err = run(cmd, cwd)
        combined = out + err
        if "ModuleNotFoundError" in combined:
            status = "FAIL ModuleNotFound"
        elif "Traceback" in combined:
            status = "FAIL traceback"
        elif rc != 0:
            status = f"FAIL rc={rc}"
        elif label == "F":
            status = "OK (server started)"
        else:
            status = "OK"
        # Show only the first line of any error for brevity.
        first_err = ""
        if rc != 0 and combined.strip():
            first_err = combined.strip().splitlines()[0][:80]
        print(f"{label:<3} {desc:<40} {rc:<4} {status}")
        if first_err:
            print(f"{'':<3} {first_err}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()