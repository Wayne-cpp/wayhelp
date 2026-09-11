from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_ch03_ddl_byte_identical():
    assert (ROOT / "sql" / "ch03-ddl.sql").read_bytes() == (
        ROOT / "db" / "init" / "03-ddl.sql"
    ).read_bytes()


def test_ch02_ddl_byte_identical():
    assert (ROOT / "sql" / "ch02-ddl.sql").read_bytes() == (
        ROOT / "db" / "init" / "01-ddl.sql"
    ).read_bytes()
