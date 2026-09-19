class AppError(Exception):
    code = "internal_error"

    def __init__(self, message: str | None = None):
        self.message = message or self.code
        super().__init__(self.message)


class MessageTooLongError(AppError):
    code = "message_too_long"


class SessionNotFoundError(AppError):
    code = "session_not_found"


class SessionCapacityReachedError(AppError):
    code = "session_capacity_reached"


class UpstreamError(AppError):
    code = "upstream_error"


class ResumeConflictError(AppError):
    """ch06(spec §9.1):无匹配挂起/ID 不符/订单不在候选内 → 409。"""
    code = "resume_conflict"
