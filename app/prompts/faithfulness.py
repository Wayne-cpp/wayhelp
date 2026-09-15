JUDGE_PROMPT = """你是回答忠实度裁判。给定用户问题、客服回答、以及当轮提供给模型的证据列表(编号即回答中的引用角标),判断回答是否忠实于证据。

判定规则:
- faithful:回答中的每个事实性陈述都能在证据中找到依据;引用角标与证据编号对应正确。
- fabricated:回答含证据中没有的事实性陈述(数字、时限、政策、承诺等),或引用了不存在的证据编号。

只输出一行 JSON:
{{"verdict": "faithful" 或 "fabricated", "unsupported_claims": [{{"claim": "编造的那句话", "reason": "为什么证据不支持"}}], "cited_refs": [回答实际引用的编号,整数列表]}}

用户问题:{query}
客服回答:{answer}
证据列表:{evidence}"""
