from dataclasses import dataclass, field
from typing import Literal

EventType = Literal["tool_call", "llm_call", "retrieval", "error"]


@dataclass
class AgentEvent:
    step_index: int
    timestamp: float
    event_type: EventType
    tool_name: str | None
    input: dict
    output: dict
    success: bool
    latency_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


@dataclass
class AgentTrace:
    trace_id: str
    agent_name: str
    task: str
    events: list[AgentEvent]

    @property
    def total_steps(self) -> int:
        return len(self.events)

    @property
    def tool_calls(self) -> list[AgentEvent]:
        return [e for e in self.events if e.event_type == "tool_call"]

    @property
    def failures(self) -> list[AgentEvent]:
        return [e for e in self.events if not e.success]

    @property
    def retrievals(self) -> list[AgentEvent]:
        return [e for e in self.events if e.event_type == "retrieval"]

    @property
    def llm_calls(self) -> list[AgentEvent]:
        return [e for e in self.events if e.event_type == "llm_call"]
