from typing import Callable, Any, Optional
from .models import ToolResult, ToolInfo

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolInfo] = {}

    def register(self, action: str, handler: Callable[[dict[str, Any]], Any], description: str = "") -> None:
        self._tools[action] = ToolInfo(
            action=action,
            description=description,
            handler=handler
        )

    def tool(self, action: str, description: str = "") -> Callable[[Callable[[dict[str, Any]], Any]], Callable[[dict[str, Any]], Any]]:
        def decorator(func: Callable[[dict[str, Any]], Any]) -> Callable[[dict[str, Any]], Any]:
            self.register(action, func, description)
            return func
        return decorator

    def dispatch_action(self, action: str, parameters: dict[str, Any]) -> ToolResult:
        if action not in self._tools:
            return ToolResult(
                success=False,
                message=f"No tool registered for action: {action}"
            )
        
        tool_info = self._tools[action]
        try:
            res = tool_info.handler(parameters)
            if isinstance(res, ToolResult):
                return res
            elif isinstance(res, str):
                return ToolResult(success=True, message=res)
            elif isinstance(res, tuple) and len(res) == 2:
                # support returning (success, message) or (success, data)
                success, val = res
                if isinstance(val, dict):
                    return ToolResult(success=success, message="Success" if success else "Failed", data=val)
                else:
                    return ToolResult(success=success, message=str(val))
            elif isinstance(res, tuple) and len(res) == 3:
                # support returning (success, message, data)
                return ToolResult(success=res[0], message=str(res[1]), data=res[2])
            elif isinstance(res, dict):
                return ToolResult(success=True, message="Completed", data=res)
            else:
                return ToolResult(success=True, message=str(res), data={"output": res})
        except Exception as e:
            return ToolResult(
                success=False,
                message=f"Error executing tool '{action}': {str(e)}"
            )

    def dispatch(self, nlu_result: Any) -> ToolResult:
        action = getattr(nlu_result, "action", None)
        parameters = getattr(nlu_result, "parameters", {}) or {}
        if not action:
            return ToolResult(success=False, message="No action in NLU result")
        return self.dispatch_action(action, parameters)

    def list_tools(self) -> list[dict[str, str]]:
        return [
            {"action": info.action, "description": info.description}
            for info in self._tools.values()
        ]

# Global default registry
default_registry = ToolRegistry()
