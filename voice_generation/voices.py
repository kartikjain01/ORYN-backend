import __init__ as config

# ==========================================================
# KOKORO VOICES
# ==========================================================

KOKORO_VOICES = {

    # ENGLISH FEMALE

    "bella": {
        "voice": "af_bella",
        "lang": "en-us",
        "label": "Bella (English Female)"
    },

    "sarah": {
        "voice": "af_sarah",
        "lang": "en-us",
        "label": "Sarah (English Female)"
    },

    "nova": {
        "voice": "af_nova",
        "lang": "en-us",
        "label": "Nova (English Female)"
    },

    "nicole": {
        "voice": "af_nicole",
        "lang": "en-us",
        "label": "Nicole (English Female)"
    },

    "sky": {
        "voice": "af_sky",
        "lang": "en-us",
        "label": "Sky (English Female)"
    },

    # ENGLISH MALE

    "michael": {
        "voice": "am_michael",
        "lang": "en-us",
        "label": "Michael (English Male)"
    },

    "adam": {
        "voice": "am_adam",
        "lang": "en-us",
        "label": "Adam (English Male)"
    },

    "echo": {
        "voice": "am_echo",
        "lang": "en-us",
        "label": "Echo (English Male)"
    },

    "eric": {
        "voice": "am_eric",
        "lang": "en-us",
        "label": "Eric (English Male)"
    },

    "liam": {
        "voice": "am_liam",
        "lang": "en-us",
        "label": "Liam (English Male)"
    },

    # HINDI FEMALE

    "alpha": {
        "voice": "hf_alpha",
        "lang": "hi",
        "label": "Alpha (Hindi Female)"
    },

    "beta": {
        "voice": "hf_beta",
        "lang": "hi",
        "label": "Beta (Hindi Female)"
    },

    # HINDI MALE

    "omega": {
        "voice": "hm_omega",
        "lang": "hi",
        "label": "Omega (Hindi Male)"
    },

    "psi": {
        "voice": "hm_psi",
        "lang": "hi",
        "label": "Psi (Hindi Male)"
    }
}

# ==========================================================
# QWEN VOICES
# ==========================================================

QWEN_VOICES = {

    "ryan": {
        "voice": "Ryan",
        "lang": "en-us",
        "label": "Ryan (English Male)"
    },

    "aiden": {
        "voice": "Aiden",
        "lang": "en-us",
        "label": "Aiden (English Male)"
    },

    "vivian": {
        "voice": "Vivian",
        "lang": "en-us",
        "label": "Vivian (English Female)"
    },

    "serena": {
        "voice": "Serena",
        "lang": "en-us",
        "label": "Serena (English Female)"
    },

    "emma": {
        "voice": "Emma",
        "lang": "en-us",
        "label": "Emma (English Female)"
    }
}

# ==========================================================
# ENGINE SELECTION
# ==========================================================

if config.ENGINE == "kokoro":

    VOICES = KOKORO_VOICES
    DEFAULT_VOICE = "michael"

elif config.ENGINE == "qwen":

    VOICES = QWEN_VOICES
    DEFAULT_VOICE = "ryan"

else:

    raise RuntimeError(f"Unknown engine: {config.ENGINE}")
