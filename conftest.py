from typing import Any, Dict, Iterator
import numpy as np
import pytest
import toml
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "tests"))

@pytest.fixture(scope="function")
def config_params() -> Iterator[Dict]:
    params: Dict[str, Any] = toml.load("./configs/params.toml")
    yield params

@pytest.fixture(scope="function")
def rng() -> Iterator[np.random.Generator]:
    rng_ = np.random.default_rng(seed=42)
    yield rng_