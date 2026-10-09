"""ch09 飞轮 prompt:标准化(口语→FAQ 式)与查重(同义归并)。输出契约单行 JSON。"""

STANDARDIZE_PROMPT = """你是电商客服知识库编辑。把用户原话改写成一条标准 FAQ 式问题,并给一条示例答案备查。

指代消解后的问题:{resolved_question}
用户原话:{raw_question}

要求:
- normalized_question:书面化、完整、不含情绪口语,不超过 60 字
- suggested_answer:基于常识的示例答案,仅供审核参考,不超过 200 字
- 只输出一行 JSON:{"normalized_question": "...", "suggested_answer": "..."}"""

DEDUP_PROMPT = """判断「新问题」与待审队列候选是否同一个意思(同义即可,不要求字面相同)。

新问题:{normalized_question}

待审队列候选(格式 id: 问题):
{candidates}

只输出一行 JSON:同义返回 {"matched_id": <候选id>},都不同返回 {"matched_id": null}
matched_id 必须来自上面候选列表,禁止发明 id。"""
