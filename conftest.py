from typing import Any, Dict, Iterator

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
