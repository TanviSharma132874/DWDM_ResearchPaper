"""Entry point: python scripts/run_pipeline.py --inspect | --all | ..."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
