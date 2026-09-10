from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.messages.utils import count_tokens_approximately

from app.errors import MessageTooLongError


def check_input_budget(system_prompt: str, current_input: str, max_input_tokens: int) -> None:
    probe = [SystemMessage(content=system_prompt), HumanMessage(content=current_input)]
    if count_tokens_approximately(probe) > max_input_tokens:
        raise MessageTooLongError("current input exceeds input token budget")
