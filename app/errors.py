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
