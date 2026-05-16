from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import sys
import os
from pathlib import Path
import glob
from datetime import datetime
from typing import Optional

# Добавим текущую директорию в путь, чтобы можно было импортировать функции
sys.path.append(str(Path(__file__).parent.parent))

from src.generate import generate

app = FastAPI(
    title="MIDIsynthesis API",
    description="API for generating music using MIDI Transformer model",
    version="0.1.0"
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/generate")
async def generate_music(seed: Optional[int] = None):
    """
    Generate music using MIDI Transformer model

    Parameters:
    - seed: Optional[int] - Random seed for reproducible generation

    Returns:
    - Dictionary with status and path to generated MIDI file
    """
    try:
        # Вызов функции генерации
        generate(task=None, seed=seed)

        # Найти последний сгенерированный файл
        list_of_files = glob.glob('Data/outputs/generated_*.mid')
        if not list_of_files:
            raise HTTPException(status_code=404, detail="No MIDI file generated")

        latest_file = max(list_of_files, key=os.path.getctime)

        return {
            "status": "success",
            "message": "Music generated successfully",
            "file_path": latest_file,
            "download_url": f"/api/download/{os.path.basename(latest_file)}",
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating music: {str(e)}"
        )

@app.get("/api/download/{filename}")
async def download_midi(filename: str):
    """
    Download generated MIDI file

    Parameters:
    - filename: str - Name of the MIDI file to download

    Returns:
    - MIDI file as FileResponse
    """
    file_path = f"Data/outputs/{filename}"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="audio/midi",
        filename=filename
    )

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "MIDIsynthesis API",
        "version": "0.1.0",
        "endpoints": {
            "/api/generate": "POST - Generate music",
            "/docs": "GET - Swagger documentation",
            "/redoc": "GET - ReDoc documentation"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)