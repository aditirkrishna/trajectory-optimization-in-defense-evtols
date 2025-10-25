import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    env_file = repo_root / "environment.yml"
    if not env_file.exists():
        print("environment.yml not found", file=sys.stderr)
        return 2
    print("To create the conda environment, run:")
    print(f"  conda env create -f {env_file} && conda activate evtol")
    print("Then install local packages (optional):")
    print("  pip install -e perception-layer -e planning-layer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


