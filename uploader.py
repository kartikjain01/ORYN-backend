import os
from dotenv import load_dotenv
from supabase import create_client

# 1. Load Environment Variables
load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_SERVICE_KEY")

if not url or not key:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_SERVICE_KEY in .env file")

supabase = create_client(url, key)

def upload_voice(user_id, file_path):
    """
    Uploads a voice file and returns the public URL for the frontend to use.
    """
    if not os.path.exists(file_path):
        print(f"❌ Error: Local file '{file_path}' not found.")
        return None

    file_name = os.path.basename(file_path)
    # Organization: voice-assets/user_123/output.wav
    storage_path = f"{user_id}/{file_name}"
    
    with open(file_path, 'rb') as f:
        try:
            # .upload() with upsert=True allows replacing the file if it exists
            supabase.storage.from_("voice-assets").upload(
                path=storage_path,
                file=f,
                file_options={
                    "content-type": "audio/wav",
                    "x-upsert": "true" 
                }
            )
            
            # Generate the Public URL so your website can play the audio
            public_url = supabase.storage.from_("voice-assets").get_public_url(storage_path)
            
            print(f"✅ Uploaded successfully!")
            print(f"🔗 Public URL: {public_url}")
            return public_url

        except Exception as e:
            print(f"❌ Upload failed: {e}")
            return None

# --- EXECUTION ---
if __name__ == "__main__":
    # Ensure you have a 'test.wav' file in your 'backend' folder before running
    test_user = "test_user_001"
    test_file = "test.wav"
    
    result_url = upload_voice(test_user, test_file)
    
    if result_url:
        print(f"\nYour file is ready at: {result_url}")