"""FastAPI app to serve the trained heart disease prediction model.

Endpoints:
- GET / -> redirect to menu.html
- GET /favicon.ico -> favicon
- POST /predict -> single patient prediction
- POST /predict/batch -> batch prediction for CSV data
- POST /predict/image -> image-based prediction using CNN
"""
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io


MODEL_PATH = Path("models/best_model.joblib")
CNN_MODEL_PATH = Path("models/heart_disease_cnn.pt")
app = FastAPI(title="Heart Disease Prediction API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (web pages)
WEB_PATH = Path("web")
if WEB_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_PATH)), name="static")
else:
    print("⚠️  Warning: web directory not found")

# Global variables
model = None
cnn_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ImagePredictRequest(BaseModel):
    image_base64: str
    image_type: Optional[str] = "jpg"


class Patient(BaseModel):
    Age: float
    Gender: str
    Weight: float
    Height: float
    BMI: float
    Smoking: str
    Alcohol_Intake: str
    Physical_Activity: str
    Diet: str
    Stress_Level: str
    Hypertension: int
    Diabetes: int
    Hyperlipidemia: int
    Family_History: int
    Previous_Heart_Attack: int
    Systolic_BP: float
    Diastolic_BP: float
    Heart_Rate: float
    Blood_Sugar_Fasting: float
    Cholesterol_Total: float


class BatchPredictRequest(BaseModel):
    data: List[dict]


@app.on_event("startup")
def load_model():
    global model, cnn_model
    
    # Load scikit-learn model
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print("✅ Scikit-learn model loaded successfully")
    
    # Load CNN model (optional)
    if CNN_MODEL_PATH.exists():
        try:
            cnn_model = models.resnet50(pretrained=False)
            cnn_model.fc = nn.Linear(cnn_model.fc.in_features, 2)
            checkpoint = torch.load(CNN_MODEL_PATH, map_location=device)
            
            # Handle different checkpoint formats
            state_dict = None
            if isinstance(checkpoint, dict):
                if "model_state_dict" in checkpoint:
                    state_dict = checkpoint["model_state_dict"]
                elif "state_dict" in checkpoint:
                    state_dict = checkpoint["state_dict"]
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Handle backbone wrapper (e.g., model.backbone.*)
            if state_dict and any(k.startswith("backbone.") for k in state_dict.keys()):
                state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items() 
                             if k.startswith("backbone.")}

            
            if state_dict:
                cnn_model.load_state_dict(state_dict, strict=False)
                cnn_model.to(device)
                cnn_model.eval()
                print("✅ CNN model loaded successfully")
            else:
                print(f"⚠️  CNN model: no state_dict found")
                cnn_model = None
        except Exception as e:
            print(f"⚠️  CNN model loading failed: {e}")
            cnn_model = None
    else:
        print(f"⚠️  CNN model not found at {CNN_MODEL_PATH}")


@app.get("/")
def read_root():
    """Redirect to menu.html"""
    return RedirectResponse(url="/menu.html", status_code=301)


@app.get("/menu.html", include_in_schema=False)
def get_menu():
    """Serve menu.html"""
    menu_path = Path("web/menu.html")
    if menu_path.exists():
        return FileResponse(menu_path, media_type="text/html")
    return HTMLResponse("<h1>Menu not found</h1>", status_code=404)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve favicon.ico"""
    favicon_path = Path("web/favicon.ico")
    if favicon_path.exists():
        return FileResponse(favicon_path, media_type="image/x-icon")
    # Fallback: return a minimal favicon if file doesn't exist
    return FileResponse(Path("web/favicon.ico"), media_type="image/x-icon")


@app.post("/predict")
def predict(p: Patient):
    """Single patient prediction"""
    try:
        row = pd.DataFrame([p.dict()])
        probs = model.predict_proba(row)[:, 1]
        score = float(probs[0])
        label = "High" if score >= 0.7 else ("Medium" if score >= 0.4 else "Low")
        return {"risk_score": score, "label": label, "model_version": "v1"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
def predict_batch(request: BatchPredictRequest):
    """Batch prediction for multiple patients"""
    try:
        if not request.data:
            raise ValueError("No data provided")
        
        # Convert to DataFrame
        df = pd.DataFrame(request.data)
        
        # Predict
        predictions = model.predict_proba(df)[:, 1]
        
        results = []
        for idx, (score, row) in enumerate(zip(predictions, request.data)):
            label = "Cao" if score >= 0.7 else ("Trung bình" if score >= 0.4 else "Thấp")
            results.append({
                "index": idx,
                "risk_score": float(score),
                "risk_probability": int(score * 100),
                "label": label,
                "age": row.get("Age", 0),
                "gender": row.get("Gender", ""),
                "systolic_bp": row.get("Systolic_BP", 0),
                "diastolic_bp": row.get("Diastolic_BP", 0),
                "cholesterol": row.get("Cholesterol_Total", 0),
                "bmi": row.get("BMI", 0),
                "heart_rate": row.get("Heart_Rate", 0),
            })
        
        return {
            "status": "success",
            "total": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    """Image-based prediction using CNN"""
    if not cnn_model:
        raise HTTPException(status_code=503, detail="CNN model not available")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Preprocess
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            outputs = cnn_model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            disease_prob = float(probabilities[0, 1].cpu().numpy())
            healthy_prob = float(probabilities[0, 0].cpu().numpy())
        
        label = "High" if disease_prob >= 0.7 else ("Medium" if disease_prob >= 0.4 else "Low")
        
        return {
            "status": "success",
            "disease_probability": disease_prob,
            "healthy_probability": healthy_prob,
            "disease_percentage": int(disease_prob * 100),
            "healthy_percentage": int(healthy_prob * 100),
            "label": label,
            "confidence": max(disease_prob, healthy_prob),
            "model": "CNN (ResNet50)"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image prediction error: {str(e)}")


@app.get("/data/{filename}", include_in_schema=False)
def serve_data(filename: str):
    """Serve CSV files from data directory"""
    data_path = Path(f"data/{filename}")
    if data_path.exists() and filename.endswith(".csv"):
        return FileResponse(data_path, media_type="text/csv")
    return HTMLResponse("<h1>Data file not found</h1>", status_code=404)


@app.get("/{page}", include_in_schema=False)
def serve_page(page: str):
    """Serve HTML pages from web directory"""
    if page.endswith(".html"):
        page_path = Path(f"web/{page}")
        if page_path.exists():
            return FileResponse(page_path, media_type="text/html")
    return HTMLResponse("<h1>Page not found</h1>", status_code=404)
