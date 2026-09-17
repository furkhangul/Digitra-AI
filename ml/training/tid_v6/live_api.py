"""Local V6 camera test bridge using Digitra's existing recognition session.

Run from the project root with the API Python environment:
  python -m uvicorn live_api:app --app-dir ml/training/tid_v6 --host 127.0.0.1 --port 8006
Camera crops are processed in memory and never saved.
"""
import asyncio
import hashlib
import json
import sys
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'apps/api'))

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from app.services.inference import DigitraLandmarkClassifier
from live_session import V6RecognitionSession

MODEL = ROOT / 'models/digitra-tid-augmented-v6'
MODEL_ID = 'digitra-tid-augmented'
VERSION = '6.0.0'
RECOGNITION_REVISION = 'g-n-1'
ORIGINS = ['http://localhost:3000', 'http://127.0.0.1:3000']


@lru_cache
def classifier():
    locked = json.loads((MODEL / 'test_started.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256((MODEL / 'model.pt').read_bytes()).hexdigest()
    if digest != locked['model_sha256']:
        raise RuntimeError('V6 checkpoint differs from the evaluated model')
    torch.set_num_threads(4)
    return DigitraLandmarkClassifier(directory=MODEL)


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(classifier)
    yield


app = FastAPI(title='Digitra V6 local camera test', lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=['GET'])


@app.get('/health')
def health():
    model = classifier().tid_image
    return {'status': 'ok', 'recognition_revision': RECOGNITION_REVISION,
            'model': {'ready': True, 'model': MODEL_ID, 'version': VERSION,
            'device': str(model.device), 'classes': model.labels,
            'checkpoint': str(MODEL / 'model.pt')}}


@app.websocket('/ws/recognize')
async def recognize(socket: WebSocket):
    if socket.headers.get('origin') not in ORIGINS:
        await socket.close(code=1008)
        return
    await socket.accept()
    session = V6RecognitionSession(classifier())
    identity = {'model_ready': True, 'model': MODEL_ID, 'version': VERSION,
                'recognition_revision': RECOGNITION_REVISION,
                'scope': 'tid_alphabet_image'}
    await socket.send_json({**identity, 'mode': 'waiting', 'hands_detected': 0,
                            'ready': False, 'prediction': None,
                            'motion_supported': True, 'recognition_mode': 'auto'})
    try:
        while True:
            payload = await socket.receive_json()
            try:
                if not isinstance(payload, dict):
                    raise ValueError('İstek bir JSON nesnesi olmalıdır')
                # A V6 test must never silently fall back to the legacy ASL model.
                if session.recognition_mode == 'alphabet' and not payload.get('action') and payload.get('hands') and not payload.get('image'):
                    raise ValueError('V6 testi için el görüntüsü gerekli')
                result = await asyncio.to_thread(session.update, payload)
                await socket.send_json(result)
            except (KeyError, TypeError, ValueError) as error:
                await socket.send_json({**identity, 'ready': False, 'prediction': None,
                                        'error': str(error)})
    except WebSocketDisconnect:
        pass
