from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uvicorn
import base64
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

# --- CONFIGURATION ---
API_KEY_SECRET = "hackathon_26" # Must match what you submitted!
MODEL_PATH = "voice_auth_model.pth"

app = FastAPI()

# --- 1. LOAD THE BRAIN (AI MODEL) ---
# We must define the same structure (ResNet18) to load the weights
device = torch.device("cpu") # Render uses CPU
print("🧠 Loading AI Model...")

try:
    model = models.resnet18(pretrained=False) # No need to download ImageNet weights again
    model.fc = nn.Linear(model.fc.in_features, 2) # Match our 2 output classes (Fake/Real)
    
    # Load the weights you trained
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval() # Set to evaluation mode
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ CRITICAL ERROR LOADING MODEL: {e}")
    model = None

# Transform pipeline (Must match training!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- HELPER: AUDIO TO SPECTROGRAM ---
def process_audio(audio_bytes):
    # Save temp file
    temp_filename = "temp_audio.mp3"
    with open(temp_filename, "wb") as f:
        f.write(audio_bytes)
    
    try:
        # Load audio (Resample to 16k to match training)
        y, sr = librosa.load(temp_filename, sr=16000, duration=3.0)
        
        # Generate Spectrogram
        plt.figure(figsize=(2.24, 2.24), dpi=100)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr)
        plt.axis('off')
        
        # Save to buffer (memory) instead of disk to be faster
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        return Image.open(buf).convert('RGB')
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# --- API ENDPOINTS ---
class AudioPayload(BaseModel):
    audio_base64: str
    language: str = "English"

@app.get("/")
def home():
    return {"status": "Spectral-ResNet is Active", "model_loaded": model is not None}

@app.post("/detect")
async def detect_voice(payload: AudioPayload, x_api_key: str = Header(None)):
    if x_api_key != API_KEY_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API Key")

    try:
        # 1. Decode Base64
        audio_bytes = base64.b64decode(payload.audio_base64)
        
        # 2. Process to Image
        spectrogram_image = process_audio(audio_bytes)
        
        # 3. AI Prediction
        if model:
            image_tensor = transform(spectrogram_image).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                
                # Assuming Class 0 = Fake, Class 1 = Real (Check your training labels!)
                # Usually ImageFolder sorts alphabetically: 'fake' comes before 'real'
                # So Index 0 = Fake, Index 1 = Real
                ai_score = probabilities[0][0].item() 
                real_score = probabilities[0][1].item()
                
                # Logic: If AI score is higher, it's AI
                is_ai = ai_score > real_score
                confidence = ai_score if is_ai else real_score
                
                return {
                    "classification": "AI_GENERATED" if is_ai else "HUMAN",
                    "confidence_score": round(confidence, 4)
                }
        else:
            # Fallback if model failed to load
            return {"classification": "HUMAN", "confidence_score": 0.5, "error": "Model offline"}

    except Exception as e:
        print(f"Error: {e}")
        return {"classification": "HUMAN", "confidence_score": 0.0, "error": str(e)}