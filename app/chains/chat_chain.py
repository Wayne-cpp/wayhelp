from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    trim_messages,
)
from langchain_core.messages.utils import count_tokens_approximately

from app.errors import MessageTooLongError
from app.sessions import StoredMessage


def check_input_budget(system_prompt: str, current_input: str, max_input_tokens: int) -> None:
    probe = [SystemMessage(content=system_prompt), HumanMessage(content=current_input)]
    if count_tokens_approximately(probe) > max_input_tokens:
        raise MessageTooLongError("current input exceeds input token budget")


def build_chat_messages(
    system_prompt: str,
    history: list[StoredMessage],
    current_input: str,
    max_input_tokens: int,
) -> list[BaseMessage]:
    check_input_budget(system_prompt, current_input, max_input_tokens)
    system = SystemMessage(content=system_prompt)
    history_msgs: list[BaseMessage] = [
        HumanMessage(content=m.content) if m.role == "user" else AIMessage(content=m.content)
        for m in history
    ]
    current = HumanMessage(content=current_input)
    trimmed = trim_messages(
        [system, *history_msgs, current],
        max_tokens=max_input_tokens,
        token_counter=count_tokens_approximately,
        strategy="last",
        include_system=True,
        start_on="human",
        end_on="human",
        allow_partial=False,
    )
    _validate_trimmed(trimmed, current_input, max_input_tokens)
    return trimmed


def _validate_trimmed(messages: list[BaseMessage], current_input: str, max_input_tokens: int) -> None:
    if not messages or not isinstance(messages[0], SystemMessage):
        raise RuntimeError("trimmed messages lost the system prompt")
    if not isinstance(messages[-1], HumanMessage) or messages[-1].content != current_input:
        raise RuntimeError("trimmed messages lost the current human message")
    rest = messages[1:]
    if not isinstance(rest[0], HumanMessage):
        raise RuntimeError("trimmed messages must start with a human message after system")
    for a, b in zip(rest, rest[1:]):
        if type(a) is type(b):
            raise RuntimeError("trimmed messages must alternate human/assistant")
    if count_tokens_approximately(messages) > max_input_tokens:
        raise RuntimeError("trimmed messages still exceed input token budget")
