from dataclasses import dataclass, field
from typing import Callable, Any

@dataclass
class ToolResult:
    success: bool
    message: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "message": self.message,
            "data": self.data
        }

@dataclass
class ToolInfo:
    action: str
    description: str
    handler: Callable[[dict[str, Any]], Any]
