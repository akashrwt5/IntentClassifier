# Intent Specifications Summary

60 intents. Provenance: 59 × assistant-session (claude-opus-5), pending human review, 1 × human (from blueprint, for privacy).

> **Review these before Stage 1.** They are the source of truth for every
> downstream label, so an error here is multiplied by the per-intent
> generation budget.

| Intent | Family | Business description | Positive | Hard negative |
|---|---|---|---|---|
| `Cmd.ActivityAerobics` | ActivityTracking | Report the user's aerobic activity data - duration, progress toward the aerobics goal, or calories burned doing aerobics. | Have I reached my aerobics goal? | How do I change my aerobic goal? |
| `Cmd.ActivityCalories` | ActivityTracking | Report total calories burned over a period when NO specific activity is named. | How many calories have I burned today? | How many calories have I burned while biking? |
| `Cmd.ActivityCycle` | ActivityTracking | Report the user's cycling or biking data - distance, duration, progress toward the cycling goal, or calories burned cycling. | How far have I cycled this week? | Where can I see the distance that I biked? |
| `Cmd.ActivityExercise` | ActivityTracking | Report the user's general exercise or workout data when no specific tracked activity is named - duration, progress toward the exercise goal, or calories burned working out. | How much longer do I need to work out today? | How do I set up my exercise goal? |
| `Cmd.ActivityRun` | ActivityTracking | Report the user's running or jogging data - distance, duration, progress toward the running goal, or calories burned running. | How much more running do I need to do today? | How do I set my running goal? |
| `Cmd.ActivityStand` | ActivityTracking | Report the user's standing data - time spent standing, progress toward the stand goal, or calories burned standing. | How close am I to my standing goal? | How do I change my stand goal? |
| `Cmd.ActivityStep` | ActivityTracking | Report the user's step count data - steps taken, progress toward the step goal, distance covered, or calories burned from steps. | How many more steps do I need today? | How do I set up my step goal? |
| `Cmd.ActivityWalk` | ActivityTracking | Report the user's walking data - distance, duration, progress toward the walking goal, or calories burned walking. | How much have I walked this week? | How do I change my walking goal? |
| `Cmd.VolumeDecrease` | AudioControl | Lower the amplification of the hearing aids, either on both sides or on one named side. The user wants sound to be QUIETER but still audible. | That's a bit much, bring it down a little. | It sounds tinny. |
| `Cmd.VolumeIncrease` | AudioControl | Raise the amplification of the hearing aids, either on both sides or on one named side. The user wants sound to be LOUDER. | Could you make my hearing aids a bit louder please? | How do I make my hearing aids louder? |
| `Cmd.VolumeMute` | AudioControl | Silence the hearing aids completely, on both sides or on one named side. The endpoint is no sound, not merely low sound. | Silence my hearing aids for a moment. | Stop streaming from the TV. |
| `Cmd.VolumeUnmute` | AudioControl | Restore sound after the hearing aids have been muted, on both sides or on one named side. | Turn my hearing aids back on, please. | Turn on the TV streaming. |
| `Cmd.FindMyPhone` | DeviceLocate | Help the user locate their misplaced PHONE using the hearing aids. | I've mislaid my phone again, can you find it? | I can't find my left hearing aid. |
| `Cmd.BatteryLevel` | DeviceStatus | Report the current remaining battery level of the hearing aids. | How much battery have I got left? | How do I charge my hearing aid? |
| `Cmd.EdgeModeDeactivate` | EdgeMode | Switch Edge Mode off entirely and return the hearing aids to their previous program. | That's enough, cancel edge mode. | Give me a little less edge mode. |
| `Cmd.EdgeModeDecrease` | EdgeMode | Reduce the strength of Edge Mode processing while leaving it active. The user finds the current processing excessive or unnatural. | There's too much edge mode, ease it back a bit. | Turn edge mode off completely. |
| `Cmd.EdgeModeIncrease` | EdgeMode | Apply or strengthen Edge Mode - the adaptive processing that improves speech clarity in difficult listening environments. The user is struggling to UNDERSTAND, not merely to hear loudly enough. | I can't follow what she's saying, it's so noisy in here. | How do I improve my hearing with Edge Mode? |
| `Default Fallback Intent` | Fallback | Catch-all for any utterance the assistant must not act on: input that is out of scope for the hearing aids, and in-scope observations that describe a state without requesting a change. Routing here produces a clarification rather than a device action, so it is the safe default whenever intent is uncertain. | The left one has been sounding a bit quiet today. | It's quieter today, make it louder. |
| `Help_AppSettings` | HelpAppSettings | Explain the app's own settings - where they are, how to change them, and the difference between Basic and Advanced app modes. | What's the difference between Basic and Advanced mode? | What's the firmware version of my aid? |
| `Help_Customize` | HelpAppSettings | Explain how to fine-tune the SOUND of the hearing aids - the equalizer, bass and treble, background, wind and machine noise, and speech focus. | How do I use the equalizer to turn down machine noise? | It's too noisy in here, help me hear him. |
| `Help_DemoMode` | HelpAppSettings | Explain demo mode - using or demonstrating the app without hearing aids connected. | Can I use the app without hearing aids connected? | Why won't my hearing aids connect? |
| `Help_DeviceSettings` | HelpAppSettings | Explain hearing-aid hardware settings - firmware and serial number, double tap sensitivity, notification playback, automatic streaming, Comfort Boost, automatic telephone programs and data logging. | The double tap is too sensitive, can I adjust it? | How do I access the app settings? |
| `Help_Home` | HelpAppSettings | Explain the Home or main screen - how to reach it, what it shows, and what can be adjusted from it. | What's on the Home screen? | Where do I find the app settings? |
| `Help_WhatsNew` | HelpAppSettings | Explain what has changed in the current version of the app or hearing aids, and provide the quick start or getting-started overview. | What's new in this version of the app? | Which version of the app am I using? |
| `Help_EdgeMode` | HelpAudio | Explain what Edge Mode is and how to use it to improve hearing in difficult environments. | What does Edge Mode actually do? | Use edge mode to hear the speaker better. |
| `Help_IntelliVoice` | HelpAudio | Explain the IntelliVoice feature - what it is, what it does, when to use it, and where to find it. | When should I use IntelliVoice? | How does the voice assistant work? |
| `Help_MaskMode` | HelpAudio | Explain Mask Mode - the setting that compensates for speech muffled by face masks - including how to switch it on or off. | How do I turn Mask Mode on? | Add mask as a customized memory. |
| `Help_Tinnitus` | HelpAudio | Explain the tinnitus masker feature - the therapeutic noise or white noise the aids can play - and how to find and adjust its settings. | How do I turn up the tinnitus noise? | Turn up the volume in my hearing aids. |
| `Help_Volume` | HelpAudio | Explain how to control hearing-aid volume - adjusting, muting, unmuting, per-side control - and help with volume problems. | How do I turn the volume down on just one aid? | Turn the volume down on my left aid. |
| `Help_HearShare` | HelpConnectivity | Explain HearShare - sharing hearing-aid, activity and wellness data with a family member or carer - including invitations and adding people. | How do I share my hearing information with my daughter? | How do I connect to my audiologist? |
| `Help_Pairing` | HelpConnectivity | Explain how to pair, connect, disconnect, sync or troubleshoot the Bluetooth connection between the hearing aids and a phone, tablet or car. | Why won't my hearing aids sync with my phone? | How do I unpair my Remote Mic? |
| `Help_RemoteProgramming` | HelpConnectivity | Explain remote programming, also branded Hearing Care Anywhere or TeleHear - how a hearing professional adjusts the aids remotely, how to request an adjustment, and the cloud account it requires. | Can my audiologist adjust my aids remotely? | How do I customize my hearing aid settings myself? |
| `Help_Accessories` | HelpDeviceCare | Explain hearing-aid accessories - TV streamer, remote microphone and others - including pairing, renaming, removing them, and controlling their volume. | How do I rename my TV streamer? | Start streaming from the TV. |
| `Help_Battery` | HelpDeviceCare | Explain how to charge a rechargeable hearing aid or change a disposable battery, including for specific device styles. | How do I change the battery in my hearing aid? | What is my battery level? |
| `Help_CleanCare` | HelpDeviceCare | Explain how to clean and care for hearing aids, including which tools and substances are safe, and how to deal with earwax build-up. | How do I get rid of wax build-up in my hearing aid? | Why doesn't my right hearing aid work? |
| `Help_InsertDevice` | HelpDeviceCare | Explain how to physically put hearing aids into the ears and take them out, including how a correctly seated aid should look. | Show me how to put my hearing aid in. | How do I clean my hearing aid? |
| `Help_SelfCheck` | HelpDeviceCare | Explain the self-check diagnostic and help with hearing aids that are faulty, not detected, or performing poorly. | There's a problem with my left hearing aid. | My hearing aids are too quiet, turn them up. |
| `Help_WiCROS` | HelpDeviceCare | Explain CROS and WiCROS systems - the transmitter and receiver pair used for single-sided hearing - and the balance control between them. | How does the CROS balance feature work? | How do I change the volume on my hearing aids? |
| `Help_FindMyHearingAids` | HelpFind | Help the user locate misplaced HEARING AIDS, including one specific side. | I've misplaced my left hearing aid. | I can't find my phone. |
| `Help_Activity` | HelpHealth | Explain how to set up, change or edit activity goals - steps, walking, running, biking, aerobics - and where activity distance is shown. | How do I change my walking goal? | How far have I walked today? |
| `Help_FallAlert` | HelpHealth | Explain the fall detection and alert feature - how it works, how to set it up, how alert contacts are managed, and what the user will see or hear when an alert is sent or cancelled. | How do I set up fall detection? | Where are my hearing aids? |
| `Help_Health` | HelpHealth | Explain the Health screen - what it shows, how to view health and hearing goals, and how to change the period those goals are measured over. | What does the Health screen show? | How do I set my step goal? |
| `Help_HeartRate` | HelpHealth | Explain the heart rate feature - what it is, how it works, how to measure heart rate, and where to find it in the app. | Where do I find my heart rate? | What is a normal heart rate recovery number? |
| `Help_HeartRateRecovery` | HelpHealth | Explain heart rate recovery - what the measurement means, how it is calculated, how to access it, and what a good value looks like. | How is heart rate recovery calculated? | How do I measure my heart rate? |
| `Help_ThriveScore` | HelpHealth | Explain the Thrive wellness score and its component scores - Body, Brain, Kind and iPro - including how they are calculated and how to improve them. | How is the Thrive score calculated? | How do I see my health goals? |
| `Help_Transcribe` | HelpSpeechServices | Explain the Transcribe feature - live speech-to-text captioning - what it does, how it works, where to find it, and whether transcripts are saved. | Does transcribe save my conversations? | Start transcribing this conversation. |
| `Help_Translate` | HelpSpeechServices | Explain the Translate feature - how it works, which languages are supported, where to find it, and whether conversations are saved. | What translation languages are available? | How do I say good morning in Russian? |
| `Cmd.MemoryChange` | Memories | Switch the hearing aids to a different saved memory or program - Normal, Restaurant, Outdoors, Car, Music, Television and the rest - either by naming it directly or by describing the listening environment the user has just entered. | I've just sat down in a busy restaurant. | How do I change to a different memory? |
| `Help_ChangingMemories` | Memories | Explain how to switch between existing hearing aid memories or programs, using the app or the button on the aid. | How do I switch to the Music memory? | How do I create a custom memory? |
| `Help_MemoryOptions` | Memories | Explain how to create, save, personalize, geotag, reset and delete hearing aid memories. | Can I create my own custom memory? | How do I change to a different memory? |
| `Cmd.ListenMessage` | Messaging | Play back or read out messages the user has received - voice or push-to-talk messages, and incoming texts read aloud. | Read me the latest message. | Send a message to my son. |
| `Cmd.SendMessage` | Messaging | Record and send a voice or push-to-talk message, optionally to a named recipient. | Let my daughter know I'm on my way. | Play my last message. |
| `Help_VoiceAssistant` | Messaging | Explain the Thrive voice assistant itself - what it is, what it can be asked to do, how to reach it, and whether it works without a phone. | What can I ask Thrive Assistant to do? | What is IntelliVoice? |
| `Help_Reminder` | Reminders | Explain the reminders feature - what it is, whether it is available, and how to create or manage reminders. | How do I set up a daily reminder? | Set a daily reminder for my tablets. |
| `reminders.add` | Reminders | Create a new reminder, optionally with a subject, a time, a date or a recurrence. | Don't let me forget to pick up milk today. | Can I set reminders in my hearing aids? |
| `reminders.complete` | Reminders | Mark an existing reminder as done. | Mark my last reminder as done. | Delete that reminder. |
| `Cmd.TranscribeStart` | SpeechServices | Begin live speech-to-text transcription of the current conversation. | Write down this conversation for me. | What does transcribe do? |
| `Cmd.TranslationStart` | SpeechServices | Begin translation, either by starting the translation feature for a named language or by asking for a specific phrase to be translated. | How do I say take me to the hotel in Spanish? | What languages can the translate feature use? |
| `Cmd.StreamingStart` | Streaming | Begin routing audio from an external source - a TV streamer or a remote microphone - directly into the hearing aids. | Let me hear the TV through my aids. | How do I start streaming from my TV streamer? |
| `Cmd.StreamingStop` | Streaming | Stop routing external audio into the hearing aids and return them to ambient listening. | I'm done with the TV, disconnect it. | Mute my hearing aids. |

## Neighbour graph

| Intent | Neighbours |
|---|---|
| `Cmd.ActivityAerobics` | Cmd.ActivityExercise, Cmd.ActivityCalories, Help_Activity, Help_Health, Default Fallback Intent |
| `Cmd.ActivityCalories` | Cmd.ActivityExercise, Cmd.ActivityWalk, Cmd.ActivityRun, Help_Activity, Help_Health, Default Fallback Intent |
| `Cmd.ActivityCycle` | Cmd.ActivityRun, Cmd.ActivityWalk, Cmd.ActivityExercise, Help_Activity, Default Fallback Intent |
| `Cmd.ActivityExercise` | Cmd.ActivityAerobics, Cmd.ActivityRun, Cmd.ActivityCalories, Help_Activity, Help_Health, Default Fallback Intent |
| `Cmd.ActivityRun` | Cmd.ActivityWalk, Cmd.ActivityExercise, Cmd.ActivityCycle, Help_Activity, Default Fallback Intent |
| `Cmd.ActivityStand` | Cmd.ActivityStep, Cmd.ActivityWalk, Help_Activity, Help_Health, Default Fallback Intent |
| `Cmd.ActivityStep` | Cmd.ActivityWalk, Cmd.ActivityStand, Help_Activity, Help_Health, Default Fallback Intent |
| `Cmd.ActivityWalk` | Cmd.ActivityStep, Cmd.ActivityRun, Cmd.ActivityStand, Help_Activity, Help_Health, Default Fallback Intent |
| `Cmd.BatteryLevel` | Help_Battery, Help_SelfCheck, Default Fallback Intent |
| `Cmd.EdgeModeDeactivate` | Cmd.EdgeModeDecrease, Cmd.EdgeModeIncrease, Cmd.VolumeMute, Help_EdgeMode, Default Fallback Intent |
| `Cmd.EdgeModeDecrease` | Cmd.EdgeModeIncrease, Cmd.EdgeModeDeactivate, Cmd.VolumeDecrease, Help_EdgeMode, Default Fallback Intent |
| `Cmd.EdgeModeIncrease` | Cmd.EdgeModeDecrease, Cmd.EdgeModeDeactivate, Cmd.VolumeIncrease, Help_EdgeMode, Help_Customize, Default Fallback Intent |
| `Cmd.FindMyPhone` | Help_FindMyHearingAids, Help_Pairing, Default Fallback Intent |
| `Cmd.ListenMessage` | Cmd.SendMessage, Cmd.TranscribeStart, Cmd.StreamingStart, Help_VoiceAssistant, Default Fallback Intent |
| `Cmd.MemoryChange` | Help_ChangingMemories, Help_MemoryOptions, Cmd.EdgeModeIncrease, Help_Customize, Default Fallback Intent |
| `Cmd.SendMessage` | Cmd.ListenMessage, Cmd.TranscribeStart, reminders.add, Help_VoiceAssistant, Default Fallback Intent |
| `Cmd.StreamingStart` | Cmd.StreamingStop, Cmd.VolumeUnmute, Help_Accessories, Help_Pairing, Default Fallback Intent |
| `Cmd.StreamingStop` | Cmd.StreamingStart, Cmd.VolumeMute, Help_Accessories, Default Fallback Intent |
| `Cmd.TranscribeStart` | Cmd.TranslationStart, Help_Transcribe, reminders.add, Default Fallback Intent |
| `Cmd.TranslationStart` | Cmd.TranscribeStart, Help_Translate, Default Fallback Intent |
| `Cmd.VolumeDecrease` | Cmd.VolumeIncrease, Cmd.VolumeMute, Cmd.EdgeModeDecrease, Help_Volume, Default Fallback Intent |
| `Cmd.VolumeIncrease` | Cmd.VolumeDecrease, Cmd.VolumeUnmute, Cmd.EdgeModeIncrease, Help_Volume, Cmd.StreamingStart, Default Fallback Intent |
| `Cmd.VolumeMute` | Cmd.VolumeUnmute, Cmd.VolumeDecrease, Cmd.StreamingStop, Help_Volume, Default Fallback Intent |
| `Cmd.VolumeUnmute` | Cmd.VolumeMute, Cmd.VolumeIncrease, Cmd.StreamingStart, Help_Volume, Default Fallback Intent |
| `Default Fallback Intent` | Cmd.VolumeIncrease, Cmd.VolumeDecrease, Cmd.StreamingStart, Cmd.EdgeModeIncrease, reminders.add, Help_Volume |
| `Help_Accessories` | Cmd.StreamingStart, Cmd.StreamingStop, Help_Pairing, Help_Volume, Default Fallback Intent |
| `Help_Activity` | Help_Health, Cmd.ActivityStep, Cmd.ActivityWalk, Cmd.ActivityExercise, Default Fallback Intent |
| `Help_AppSettings` | Help_DeviceSettings, Help_WhatsNew, Help_Home, Help_Customize, Default Fallback Intent |
| `Help_Battery` | Cmd.BatteryLevel, Help_CleanCare, Help_InsertDevice, Help_SelfCheck, Default Fallback Intent |
| `Help_ChangingMemories` | Cmd.MemoryChange, Help_MemoryOptions, Help_Customize, Cmd.EdgeModeIncrease, Default Fallback Intent |
| `Help_CleanCare` | Help_Battery, Help_InsertDevice, Help_SelfCheck, Default Fallback Intent |
| `Help_Customize` | Help_MemoryOptions, Help_DeviceSettings, Help_EdgeMode, Help_Volume, Default Fallback Intent |
| `Help_DemoMode` | Help_AppSettings, Help_WhatsNew, Help_Pairing, Help_Home, Default Fallback Intent |
| `Help_DeviceSettings` | Help_AppSettings, Help_Customize, Help_MemoryOptions, Help_WiCROS, Help_Pairing, Default Fallback Intent |
| `Help_EdgeMode` | Cmd.EdgeModeIncrease, Cmd.EdgeModeDecrease, Cmd.EdgeModeDeactivate, Help_Customize, Default Fallback Intent |
| `Help_FallAlert` | Help_Health, Help_SelfCheck, Help_DeviceSettings, Default Fallback Intent |
| `Help_FindMyHearingAids` | Cmd.FindMyPhone, Help_SelfCheck, Help_Pairing, Default Fallback Intent |
| `Help_Health` | Help_Activity, Help_HeartRate, Help_ThriveScore, Help_FallAlert, Cmd.ActivityStep, Default Fallback Intent |
| `Help_HearShare` | Help_RemoteProgramming, Help_Pairing, Help_Health, Default Fallback Intent |
| `Help_HeartRate` | Help_HeartRateRecovery, Help_Health, Help_ThriveScore, Default Fallback Intent |
| `Help_HeartRateRecovery` | Help_HeartRate, Help_Health, Help_ThriveScore, Default Fallback Intent |
| `Help_Home` | Help_AppSettings, Help_WhatsNew, Help_VoiceAssistant, Default Fallback Intent |
| `Help_InsertDevice` | Help_CleanCare, Help_SelfCheck, Help_Battery, Default Fallback Intent |
| `Help_IntelliVoice` | Help_EdgeMode, Help_VoiceAssistant, Help_Customize, Default Fallback Intent |
| `Help_MaskMode` | Help_EdgeMode, Help_MemoryOptions, Help_Customize, Default Fallback Intent |
| `Help_MemoryOptions` | Help_ChangingMemories, Cmd.MemoryChange, Help_Customize, Help_MaskMode, Default Fallback Intent |
| `Help_Pairing` | Help_Accessories, Help_RemoteProgramming, Help_SelfCheck, Cmd.StreamingStart, Default Fallback Intent |
| `Help_Reminder` | reminders.add, reminders.complete, Help_Home, Default Fallback Intent |
| `Help_RemoteProgramming` | Help_Pairing, Help_Customize, Help_HearShare, Help_DeviceSettings, Default Fallback Intent |
| `Help_SelfCheck` | Help_FindMyHearingAids, Help_Pairing, Help_CleanCare, Cmd.VolumeIncrease, Default Fallback Intent |
| `Help_ThriveScore` | Help_Health, Help_HeartRate, Help_Activity, Default Fallback Intent |
| `Help_Tinnitus` | Help_Volume, Cmd.VolumeIncrease, Help_ChangingMemories, Help_Customize, Default Fallback Intent |
| `Help_Transcribe` | Cmd.TranscribeStart, Help_Translate, Help_VoiceAssistant, Default Fallback Intent |
| `Help_Translate` | Cmd.TranslationStart, Help_Transcribe, Help_VoiceAssistant, Default Fallback Intent |
| `Help_VoiceAssistant` | Help_IntelliVoice, Cmd.SendMessage, Cmd.ListenMessage, Help_Transcribe, Default Fallback Intent |
| `Help_Volume` | Cmd.VolumeIncrease, Cmd.VolumeDecrease, Cmd.VolumeMute, Help_Tinnitus, Help_Customize, Default Fallback Intent |
| `Help_WhatsNew` | Help_AppSettings, Help_Home, Help_DemoMode, Default Fallback Intent |
| `Help_WiCROS` | Help_Volume, Help_ChangingMemories, Help_DeviceSettings, Default Fallback Intent |
| `reminders.add` | reminders.complete, Help_Reminder, Cmd.TranscribeStart, Default Fallback Intent |
| `reminders.complete` | reminders.add, Help_Reminder, Default Fallback Intent |
