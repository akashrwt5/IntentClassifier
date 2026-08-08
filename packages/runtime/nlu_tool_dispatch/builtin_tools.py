from typing import Any, Optional
from .registry import default_registry
from .models import ToolResult

class DeviceState:
    def __init__(self):
        self.volume = 50
        self.muted = False
        self.memory = "Normal"
        self.streaming = False
        self.reminders = [
            {"id": 1, "name": "take medicine", "date-time": "3pm", "recurrence": "daily", "completed": False}
        ]
        self.activity_steps = 7342
        self.activity_calories = 340
        self.activity_stand_hours = 8
        self.active_session: Optional[str] = None

state = DeviceState()

# === Volume Domain ===

@default_registry.tool("volume.mute", description="Mute the hearing aids")
def mute_volume(params: dict[str, Any]) -> ToolResult:
    state.muted = True
    return ToolResult(
        success=True,
        message="Hearing aids muted. [volume → 0 (muted)]",
        data={"muted": True, "volume": state.volume}
    )

@default_registry.tool("volume.unmute", description="Unmute the hearing aids")
def unmute_volume(params: dict[str, Any]) -> ToolResult:
    state.muted = False
    return ToolResult(
        success=True,
        message=f"Hearing aids unmuted. [volume → {state.volume}]",
        data={"muted": False, "volume": state.volume}
    )

@default_registry.tool("volume.increase", description="Increase volume of the hearing aids")
def increase_volume(params: dict[str, Any]) -> ToolResult:
    state.muted = False
    old_vol = state.volume
    state.volume = min(100, state.volume + 10)
    return ToolResult(
        success=True,
        message=f"Volume increased from {old_vol} to {state.volume}.",
        data={"muted": False, "volume": state.volume}
    )

@default_registry.tool("volume.decrease", description="Decrease volume of the hearing aids")
def decrease_volume(params: dict[str, Any]) -> ToolResult:
    old_vol = state.volume
    state.volume = max(0, state.volume - 10)
    return ToolResult(
        success=True,
        message=f"Volume decreased from {old_vol} to {state.volume}.",
        data={"muted": state.muted, "volume": state.volume}
    )

# === Memory / Program Domain ===

@default_registry.tool("memory.change", description="Change hearing aid memory program")
def change_memory(params: dict[str, Any]) -> ToolResult:
    memory_name = params.get("MemoryName")
    if not memory_name:
        return ToolResult(
            success=False,
            message="Cannot change memory: MemoryName not specified."
        )
    state.memory = memory_name
    return ToolResult(
        success=True,
        message=f"Hearing aid memory program changed to '{memory_name}'.",
        data={"memory": state.memory}
    )

# === Battery Domain ===

@default_registry.tool("battery.level", description="Get battery level of hearing aids")
def get_battery(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message="Hearing aids battery status: Left 85%, Right 85%.",
        data={"left_battery": 85, "right_battery": 85}
    )

# === Find Domain ===

@default_registry.tool("phone.find", description="Locate phone by ringing it")
def find_phone(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message="Simulating Ring: Sending high-volume tone to phone...",
        data={"ringing": True}
    )

# === Streaming Domain ===

@default_registry.tool("streaming.start", description="Start audio streaming")
def start_streaming(params: dict[str, Any]) -> ToolResult:
    state.streaming = True
    return ToolResult(
        success=True,
        message="Audio streaming started from your connected device.",
        data={"streaming": True}
    )

@default_registry.tool("streaming.stop", description="Stop audio streaming")
def stop_streaming(params: dict[str, Any]) -> ToolResult:
    state.streaming = False
    return ToolResult(
        success=True,
        message="Audio streaming stopped.",
        data={"streaming": False}
    )

# === Reminders Domain ===

@default_registry.tool("reminders.add", description="Create a reminder")
def create_reminder(params: dict[str, Any]) -> ToolResult:
    name = params.get("name")
    date_time = params.get("date-time")
    recurrence = params.get("recurrence", "none")
    
    if not name or not date_time:
        return ToolResult(
            success=False,
            message=f"Cannot create reminder: missing name ('{name}') or date-time ('{date_time}')."
        )
    
    new_id = len(state.reminders) + 1
    state.reminders.append({
        "id": new_id,
        "name": name,
        "date-time": date_time,
        "recurrence": recurrence,
        "completed": False
    })
    return ToolResult(
        success=True,
        message=f"Reminder created successfully: '{name}' at {date_time}.",
        data={"reminder_id": new_id, "name": name, "date-time": date_time, "recurrence": recurrence}
    )

@default_registry.tool("reminders.complete", description="Mark a reminder as complete")
def complete_reminder(params: dict[str, Any]) -> ToolResult:
    uncompleted = [r for r in state.reminders if not r["completed"]]
    if not uncompleted:
        return ToolResult(
            success=False,
            message="No pending reminders to complete."
        )
    
    # mark the most recent uncompleted one as complete
    target = uncompleted[-1]
    target["completed"] = True
    return ToolResult(
        success=True,
        message=f"Completed reminder: '{target['name']}'.",
        data={"completed_reminder_id": target["id"]}
    )

# === Activity Domain ===

@default_registry.tool("activity.step", description="Get daily step count")
def get_steps(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message=f"You have walked {state.activity_steps} steps today.",
        data={"steps": state.activity_steps}
    )

@default_registry.tool("activity.calories", description="Get daily calories burned")
def get_calories(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message=f"You have burned {state.activity_calories} kcal today.",
        data={"calories": state.activity_calories}
    )

@default_registry.tool("activity.stand", description="Get stand goal progress")
def get_stand(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message=f"You have stand data for {state.activity_stand_hours}/12 hours today.",
        data={"stand_hours": state.activity_stand_hours}
    )

@default_registry.tool("activity.aerobics", description="Start aerobics activity tracking")
def track_aerobics(params: dict[str, Any]) -> ToolResult:
    state.active_session = "Aerobics"
    return ToolResult(success=True, message="Started tracking aerobics session.", data={"tracking": "aerobics"})

@default_registry.tool("activity.cycle", description="Start cycling activity tracking")
def track_cycle(params: dict[str, Any]) -> ToolResult:
    state.active_session = "Cycling"
    return ToolResult(success=True, message="Started tracking cycling session.", data={"tracking": "cycling"})

@default_registry.tool("activity.exercise", description="Start generic exercise tracking")
def track_exercise(params: dict[str, Any]) -> ToolResult:
    state.active_session = "Exercise"
    return ToolResult(success=True, message="Started tracking exercise session.", data={"tracking": "exercise"})

@default_registry.tool("activity.run", description="Start running activity tracking")
def track_run(params: dict[str, Any]) -> ToolResult:
    state.active_session = "Running"
    return ToolResult(success=True, message="Started tracking running session.", data={"tracking": "running"})

@default_registry.tool("activity.walk", description="Start walking activity tracking")
def track_walk(params: dict[str, Any]) -> ToolResult:
    state.active_session = "Walking"
    return ToolResult(success=True, message="Started tracking walking session.", data={"tracking": "walking"})

# === Messaging Domain ===

@default_registry.tool("message.listen", description="Listen to messages")
def listen_messages(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message="Playing your latest voice messages.",
        data={"playing": True}
    )

@default_registry.tool("message.compose", description="Compose a new message")
def compose_message(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message="Opening composer. Speak your message...",
        data={"composing": True}
    )

# === Screens/Session Domain ===

@default_registry.tool("transcribe.open", description="Start transcription screen")
def open_transcribe(params: dict[str, Any]) -> ToolResult:
    return ToolResult(success=True, message="Transcribe screen opened.", data={"screen": "transcribe"})

@default_registry.tool("translate.open", description="Start translation screen")
def open_translate(params: dict[str, Any]) -> ToolResult:
    return ToolResult(success=True, message="Translate screen opened.", data={"screen": "translate"})

# === Fallback ===

@default_registry.tool("genai.fallback", description="Forward to GenAI backend")
def genai_fallback(params: dict[str, Any]) -> ToolResult:
    return ToolResult(
        success=True,
        message="Action forwarded to GenAI fallback engine.",
        data={"fallback": True}
    )

# === Help Domain Registration ===

help_actions = [
    ("help.accessories", "accessories"),
    ("help.app_settings", "app settings"),
    ("help.battery", "battery information"),
    ("help.clean_care", "cleaning and care"),
    ("help.customize", "customizing parameters"),
    ("help.demo_mode", "Demo Mode"),
    ("help.device_settings", "device settings"),
    ("help.edge_mode", "Edge Mode"),
    ("help.fall_alert", "Fall Alert setup"),
    ("help.find_aids", "finding hearing aids"),
    ("help.health", "health tracking"),
    ("help.hear_share", "HearShare service"),
    ("help.hearing_care_connect", "Hearing Care Anywhere connection"),
    ("help.heart_rate", "Heart Rate screen"),
    ("help.heart_rate_recovery", "Heart Rate Recovery statistics"),
    ("help.home", "home screen layout"),
    ("help.insert_device", "inserting hearing aids safely"),
    ("help.intelli_voice", "IntelliVoice functionality"),
    ("help.mask_mode", "Mask Mode usage"),
    ("help.memories", "custom memories program"),
    ("help.memory_options", "memory slots configuration"),
    ("help.pairing", "pairing with Bluetooth devices"),
    ("help.reminder", "reminders and task scheduling"),
    ("help.remote_programming", "remote programming requests"),
    ("help.selfcheck", "SelfCheck diagnostic tool"),
    ("help.thrive_score", "Thrive Score calculation"),
    ("help.tinnitus", "tinnitus therapy adjustments"),
    ("help.transcribe", "Transcribe utility usage"),
    ("help.translate", "Translate utility usage"),
    ("help.voice_assistant", "Voice Assistant setup"),
    ("help.volume", "adjusting volume and balance"),
    ("help.whats_new", "latest version changes info"),
    ("help.wicros", "WiCROS compatibility and setup")
]

def make_help_handler(topic: str):
    def help_handler(params: dict[str, Any]) -> ToolResult:
        return ToolResult(
            success=True,
            message=f"Showing help screen for: {topic}.",
            data={"help_topic": topic}
        )
    return help_handler

for action, topic_desc in help_actions:
    default_registry.register(
        action=action,
        handler=make_help_handler(topic_desc),
        description=f"Display help for {topic_desc}"
    )
