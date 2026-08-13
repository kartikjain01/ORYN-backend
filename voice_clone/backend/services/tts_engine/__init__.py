#ENGINE = "xtts_v2"
ENGINE = "fish"

if ENGINE == "xtts_v2":

    from .xtts_v2 import synthesize, XTTSParams as EngineParams

elif ENGINE == "qwen":

    from .qwen import synthesize, QwenParams as EngineParams

elif ENGINE == "fish":

    from .fish import synthesize, FishParams as EngineParams

else:

    raise ValueError(f"Unsupported engine: {ENGINE}")
