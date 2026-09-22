import os
from pathlib import Path
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA numerical tests require a visible GPU")
    value = os.environ.get("TEST_CUDA_DEVICE", "cuda:0")
    torch.cuda.set_device(value)
    torch.set_num_threads(1)
    return torch.device(value)


@pytest.fixture
def frozen(device):
    import numpy as np

    with np.load(ROOT / "data/inputs/main/rep0-stream0-regime0-d10.npz", allow_pickle=False) as f:
        return {k: torch.as_tensor(f[k], device=device) for k in f.files}
