from typing import Optional, List, Dict, Any, Union

import time

import shortuuid
from pydantic import BaseModel, Field


class ChatCompletionRequest(BaseModel):
    # TODO: support streaming, stop with a list of text etc.
    model: str
    messages: List[Dict[str, Union[str, List[str], Dict[str, Union[str, int]]]]]
    temperature: Optional[float] = 0.7
    n: int = 1
    max_tokens: Optional[int] = None
    stop: Optional[str] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str
    usage: Optional[Dict[str, int]] = None  # token repartition


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=shortuuid.random)
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    choices: List[ChatCompletionResponseChoice]
    # usage: Optional[Dict[str, int]] = None # rather associate usage to the reponses choices
