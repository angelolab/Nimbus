from typing import Any, Dict, Iterator

import pytest
import toml


@pytest.fixture(scope="module")
def config_params() -> Iterator[Dict]:
    params: Dict[str, Any] = toml.load("config/params.toml")
    yield params
