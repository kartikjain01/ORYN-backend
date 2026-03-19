from fastapi import FastAPI

# Import your apps
from voice_generation.main import app as gen_app
from Voice_editor.Voice_editor.api import app as edit_app
from voice_clone.voice_clone.backend.api.main import app as clone_app

app = FastAPI()

# Mount all services
app.mount("/generate", gen_app)
app.mount("/edit", edit_app)
app.mount("/clone", clone_app)
