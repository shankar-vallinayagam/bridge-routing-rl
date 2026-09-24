from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional


@dataclass
class RoutingResult:
    case_id: str
    method: str
    success: bool
    original_cnot_count: int
    routed_cnot_count: Optional[int] = None
    added_cnot_count: Optional[int] = None
    swap_count: Optional[int] = None
    bridge_count: Optional[int] = None
    route_time_seconds: Optional[float] = None
    inference_time_seconds: Optional[float] = None
    environment_time_seconds: Optional[float] = None
    decision_count: Optional[int] = None
    depth: Optional[int] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
