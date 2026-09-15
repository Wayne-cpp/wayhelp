from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import sessionmaker


def make_engine(database_url: str) -> Engine:
    return create_engine(database_url, pool_pre_ping=True)


def make_session_factory(engine: Engine) -> sessionmaker:
    return sessionmaker(bind=engine, expire_on_commit=False)


def ping(engine: Engine) -> None:
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))


_CH04_TABLES = ("low_confidence_questions", "faith_cases")


def check_ch04_tables(engine) -> None:
    """启动只读校验:两张 ch04 表缺失即 RuntimeError,附升级命令;不 create_all。"""
    from sqlalchemy import text
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = DATABASE() AND table_name IN ('low_confidence_questions','faith_cases')"
        )).all()
    missing = [t for t in _CH04_TABLES if (t,) not in [(r[0],) for r in rows]]
    if missing:
        raise RuntimeError(
            f"缺少 ch04 表 {missing}:请执行 "
            f"docker exec -i wayhelp-mysql mysql -uroot -proot-password wayhelp < sql/ch04-ddl.sql")
