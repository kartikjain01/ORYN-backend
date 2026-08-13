from backend.core.settings import CURRENT_MODEL

if CURRENT_MODEL == "qwen":
    from backend.models.qwen import TTSModel

elif CURRENT_MODEL == "xtts":
    from backend.models.xtts import TTSModel

else:
    raise ValueError(f"Unsupported model: {CURRENT_MODEL}")


tts_model = TTSModel()
tts_model.load()
