# Ensures vendored dependencies in this repository are imported before site-packages.
# This runs very early at interpreter startup (imported by the standard 'site' module)
# as long as the repository root is on sys.path (e.g., when running from this directory).
from pathlib import Path
import sys

_repo_root = Path(__file__).resolve().parent
_vendored_pyannote = _repo_root / "pyannote"

# Prepend vendored pyannote and repo root so they shadow any site-packages installs.
if _vendored_pyannote.exists():
    sys.path.insert(0, str(_vendored_pyannote))
    sys.path.insert(0, str(_repo_root))
