from pydantic.dataclasses import dataclass


@dataclass
class LogStats:
    name: str
    num_traces: int
    num_activities: int
    num_events: int
