class TurnAbortError(Exception):
    """节点已通过 stream writer 发出 error 帧;抛出以中止图执行。
    驱动层捕获后正常结束 SSE(不发 [DONE]),不提交本轮。"""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)
