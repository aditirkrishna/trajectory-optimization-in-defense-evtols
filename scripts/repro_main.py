import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    planning_demo = repo_root / "planning-layer" / "examples" / "simple_planning_demo.py"
    if not planning_demo.exists():
        print("planning demo not found", file=sys.stderr)
        return 2
    # Execute the demo in a subprocess to capture non-zero exit codes if any
    import runpy

    try:
        runpy.run_path(str(planning_demo), run_name="__main__")
    except SystemExit as e:
        return int(getattr(e, "code", 1) or 0)
    except Exception as e:
        print(f"Demo failed: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


