from operator import add
from typing import Annotated

from langchain.agents import AgentState


class CustomAgentState(AgentState):
    initial_user_patience: float
    updated_user_patience: float
    tool_called_patience: Annotated[list[float | int], add]
