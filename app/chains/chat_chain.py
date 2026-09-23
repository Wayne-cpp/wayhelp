from app.errors import MessageTooLongError


def check_user_input(text: str, max_tokens: int) -> None:
    """ch07:用户输入 token 闸(只量输入本身,§4 尺)。"""
    from app.services.token_budget import estimate_tokens
    if estimate_tokens(text) > max_tokens:
        raise MessageTooLongError("message exceeds MAX_USER_INPUT_TOKENS")
