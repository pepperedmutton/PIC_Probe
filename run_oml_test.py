from pathlib import Path
import sys

# Ensure current directory is in path so we can import benchmarks
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import test_oml_regime, ensure_results_dir

if __name__ == "__main__":
    print("Running OML Benchmark Standalone...")
    out_dir = ensure_results_dir()
    test_oml_regime(out_dir)
    print("\nOML Benchmark Completed.")
