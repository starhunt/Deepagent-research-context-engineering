import sys
from pathlib import Path

try:
    import docker  # noqa: F401
except ImportError:
    pass

_REPO_ROOT = Path(__file__).resolve().parents[2]
_VENDORED_DEEPAGENTS = _REPO_ROOT / "deepagents_sourcecode" / "libs" / "deepagents"
if _VENDORED_DEEPAGENTS.exists() and str(_VENDORED_DEEPAGENTS) not in sys.path:
    sys.path.insert(0, str(_VENDORED_DEEPAGENTS))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
