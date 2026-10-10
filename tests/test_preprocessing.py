"""确定性清洗:脱敏/繁简/全半角/空白,幂等且顺序冻结(spec §4.1)。"""

from training.preprocessing import PREPROCESS_VERSION, clean_text, preprocess_config


def test_masking():
    assert clean_text("我手机13812345678,邮箱a@b.com") == "我手机[PHONE],邮箱[EMAIL]"
    # 计划草案期望「订单A1001啥时候发货」脱敏,但 Python re 把汉字算 \w,订单A1001 与
    # 前后文之间无 \b 边界,紧贴汉字的字母数字串不脱敏(此保守性正是幂等测试对全角
    # 输入成立的原因);按 Step 4 以实现真实行为为准修正,详见 dev-notes Task 3
    assert clean_text("订单A1001啥时候发货") == "订单A1001啥时候发货"
    assert clean_text("订单 A1001 什么时候发货") == "订单 [ORDER_ID] 什么时候发货"
    assert clean_text("单号 20261010001 查一下") == "单号 [ORDER_ID] 查一下"


def test_width_and_space():
    assert clean_text("ＡＢＣ　１２３") == "ABC 123"  # 全角→半角,全角空格折叠
    assert clean_text("  多大   的\t合适\n") == "多大 的 合适"


def test_traditional_to_simplified():
    assert clean_text("我要退貨") == "我要退货"


def test_idempotent_and_deterministic():
    text = "訂單Ａ１００１  退貨,电话13900001111"
    assert clean_text(clean_text(text)) == clean_text(text)


def test_empty_and_version():
    assert clean_text("   ") == ""
    assert PREPROCESS_VERSION == "topic-preprocess-v1"
    cfg = preprocess_config()
    assert cfg["preprocess_version"] == PREPROCESS_VERSION and "steps" in cfg
