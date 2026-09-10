import pytest
from sqlalchemy.exc import OperationalError

from app.db import make_engine, ping
from tests.conftest import make_settings


def test_ping_failure_raises():
    engine = make_engine(make_settings().database_url)  # 127.0.0.1:9 不可达
    with pytest.raises(OperationalError):
        ping(engine)
