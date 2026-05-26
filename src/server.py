from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import uvicorn
import numpy as np
from PIL import Image
import io
import os
import json
from tensorflow.keras.models import load_model

app = FastAPI(title='Crop Disease Detection')

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/crop_model.h5')
CLASS_INDICES_PATH = os.environ.get('CLASS_INDICES_PATH', 'models/class_indices.json')


def load_resources():
    model = None
    class_map = None
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        print('Model not found at', MODEL_PATH)

    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, 'r') as fh:
            class_map = json.load(fh)
    else:
        print('Class indices not found at', CLASS_INDICES_PATH)

    return model, class_map


MODEL, CLASS_MAP = load_resources()


def preprocess_image(file_bytes, img_size=224):
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize((img_size, img_size))
    arr = np.array(img).astype('float32') / 255.0
    arr = np.expand_dims(arr, 0)
    return arr


@app.post('/predict')
async def predict(file: UploadFile = File(...)):
    if MODEL is None:
        raise HTTPException(status_code=503, detail='Model not loaded')
    contents = await file.read()
    try:
        inp = preprocess_image(contents)
        preds = MODEL.predict(inp)[0]
        idx = int(np.argmax(preds))
        prob = float(preds[idx])
        label = str(idx)
        if CLASS_MAP:
            inv = {v: k for k, v in CLASS_MAP.items()}
            label = inv.get(idx, label)
        return JSONResponse({'label': label, 'probability': prob})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    uvicorn.run('src.server:app', host='0.0.0.0', port=int(os.environ.get('PORT', 8080)), reload=False)
