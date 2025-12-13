import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session")
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def triton_available():
    if not torch.cuda.is_available():
        return False
    try:
        import triton  # type: ignore  # noqa: F401
    except Exception:
        return False
    return True


@pytest.fixture(scope="session")
def cuda_extension_available():
    if not torch.cuda.is_available():
        return False
    try:
        from fa1.cuda.impl import _load_ext as _fa1_load_ext

        _fa1_load_ext()
    except Exception:
        return False
    return True
