from langchain.agents import AgentState


class CustomAgentState(AgentState):
    initial_user_patience: float
    updated_user_patience: float
