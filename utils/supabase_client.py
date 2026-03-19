import os
from supabase import create_client

# Use your Service Role Key for backend operations
url = "https://pwghoohfgjeaaumloung.supabase.co"
key = "YOUR_SERVICE_ROLE_KEY" 
supabase = create_client(url, key)

def upload_voice_asset(user_id, local_file_path):
    file_name = os.path.basename(local_file_path)
    
    # THE KEY STEP: The path must start with the user_id
    storage_path = f"{user_id}/{file_name}"
    
    with open(local_file_path, 'rb') as f:
        response = supabase.storage.from_('voice-assets').upload(
            path=storage_path,
            file=f,
            file_options={"content-type": "audio/wav"}
        )
    return response

# Usage after your AI generates a file:
# upload_voice_asset("6d3...user-uuid", "outputs/clone_01.wav")