from typing import Any, Dict, Iterator

import pytest
import toml


@pytest.fixture(scope="function")
def config_params() -> Iterator[Dict]:
    params: Dict[str, Any] = toml.load("./configs/params.toml")
    yield params
