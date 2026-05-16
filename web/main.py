from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import sys
import os
import logging
from pathlib import Path
import pretty_midi
from scipy.io import wavfile
import numpy as np
import glob
from datetime import datetime
from typing import Optional, Dict
import time

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Добавим текущую директорию в путь, чтобы можно было импортировать функции
sys.path.append(str(Path(__file__).parent.parent))

from src.generate import generate

# Логирование запуска сервера
logger.info("Starting MIDIsynthesis API server")

OUTPUT_DIR = Path("Data/outputs")
AUDIO_DIR = OUTPUT_DIR / "audio"

# Настройка кеширования
CACHE_TTL = 3600  # Время жизни кэша в секундах (1 час)
cache: Dict[str, Dict] = {}

def get_cache_key(seed: Optional[int]) -> str:
    """Генерирует ключ для кэша на основе seed"""
    return f"generate:{seed}"

def get_from_cache(seed: Optional[int]) -> Optional[Dict]:
    """Получает данные из кэша, если они не устарели"""
    cache_key = get_cache_key(seed)
    cached_data = cache.get(cache_key)

    if cached_data:
        # Проверяем, не устарели ли данные в кэше
        if time.time() - cached_data['timestamp'] < CACHE_TTL:
            logger.info(f"Cache hit for seed: {seed}")
            return cached_data
        else:
            # Данные устарели, удаляем их из кэша
            cache.pop(cache_key, None)
            logger.info(f"Cache expired for seed: {seed}")

    return None

def add_to_cache(seed: Optional[int], data: Dict):
    """Добавляет данные в кэш"""
    cache_key = get_cache_key(seed)
    data_with_timestamp = {
        **data,
        'timestamp': time.time(),
        'from_cache': False
    }
    cache[cache_key] = data_with_timestamp
    logger.info(f"Added to cache for seed: {seed}")

def convert_midi_to_wav(midi_path: str) -> Path:
    midi_path = Path(midi_path)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    wav_path = AUDIO_DIR / f"{midi_path.stem}.wav"

    midi_data = pretty_midi.PrettyMIDI(str(midi_path))

    # Сделать все инструменты фортепиано
    for instrument in midi_data.instruments:
        instrument.program = 0          # Acoustic Grand Piano
        instrument.is_drum = False

    audio = midi_data.synthesize(fs=44100)

    if len(audio) == 0:
        raise RuntimeError("Generated WAV is empty")

    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs

    audio_int16 = np.int16(audio * 32767)
    wavfile.write(str(wav_path), 44100, audio_int16)

    logger.info(f"Converted MIDI to WAV with piano sound: {wav_path}")

    return wav_path

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

# Настройка статических файлов
app.mount("/static", StaticFiles(directory="web/static"), name="static")

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
        logger.info(f"Starting music generation with seed: {seed}")

        # Проверяем кэш только если seed явно задан
        if seed is not None:
            cached_result = get_from_cache(seed)
            if cached_result:
                logger.info(f"Returning cached result for seed: {seed}")
                return {
                    **cached_result,
                    "from_cache": True,
                    "message": "Music generated successfully (from cache)"
                }

        # Вызов функции генерации
        generate(task=None, seed=seed)

        # Найти последний сгенерированный файл
        list_of_files = glob.glob('Data/outputs/generated_*.mid')
        if not list_of_files:
            error_msg = "No MIDI file generated"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)

        latest_file = max(list_of_files, key=os.path.getctime)
        wav_file = convert_midi_to_wav(latest_file)
        logger.info(f"Successfully generated music: {latest_file}")

        # Сохраняем результат в кэш
        result = {
            "status": "success",
            "message": "Music generated successfully",
            "file_path": latest_file,
            "download_url": f"/api/download/{os.path.basename(latest_file)}",
            "audio_url": f"/api/audio/{wav_file.name}",
            "timestamp": datetime.now().isoformat()
        }

        if seed is not None:
            add_to_cache(seed, result)
            
        return result

    except Exception as e:
        logger.error(f"Error generating music: {str(e)}", exc_info=True)
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
    logger.info(f"Attempting to download file: {filename}")

    if not os.path.exists(file_path):
        error_msg = f"File not found: {file_path}"
        logger.error(error_msg)
        raise HTTPException(status_code=404, detail=error_msg)

    logger.info(f"Successfully downloaded file: {filename}")
    return FileResponse(
        file_path,
        media_type="audio/midi",
        filename=filename
    )

@app.get("/api/audio/{filename}")
async def play_audio(filename: str):
    safe_filename = Path(filename).name
    file_path = AUDIO_DIR / safe_filename

    logger.info(f"Attempting to play audio file: {file_path}")

    if not file_path.exists():
        error_msg = f"Audio file not found: {file_path}"
        logger.error(error_msg)
        raise HTTPException(status_code=404, detail=error_msg)

    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=safe_filename
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