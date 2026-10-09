"""ch09 Langfuse 配置契约:默认关闭;密钥齐才视为可用。"""
from app.config import Settings


def test_langfuse_disabled_by_default():
    s = Settings(_env_file=None, openai_base_url="http://x", openai_api_key="k",
                 model_name="m", database_url="mysql+pymysql://u:p@h/d")
    assert s.langfuse_enabled is False
    assert s.has_langfuse_key() is False


def test_langfuse_key_detection():
    s = Settings(_env_file=None, openai_base_url="http://x", openai_api_key="k",
                 model_name="m", database_url="mysql+pymysql://u:p@h/d",
                 langfuse_enabled=True, langfuse_public_key="pk-lf-x",
                 langfuse_secret_key="sk-lf-x")
    assert s.has_langfuse_key() is True
    assert s.langfuse_host == "http://localhost:3000"


def test_compose_has_langfuse_stack():
    text = open("docker-compose.yml", encoding="utf-8").read()
    for svc in ("langfuse-web:", "langfuse-worker:", "clickhouse:",
                "langfuse-postgres:", "valkey:", "minio:"):
        assert svc in text, svc
    assert "127.0.0.1:3000:3000" in text
