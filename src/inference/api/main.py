import io
import glob
import cv2
from PIL import Image

import numpy as np
import tensorflow as tf

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse, StreamingResponse


app = FastAPI(title="Airbus-Ship-Detection")


@app.on_event("startup")
def load_model():
    # Load the model
    global model
    model_path = "./ouputs/models/unet_*.h5"
    # Find the path matching the pattern
    model_files = glob.glob(model_path)

    if not model_files:
        raise FileNotFoundError("No model files found matching the pattern")

    model = tf.keras.models.load_model(model_files[0])


# redirect
@app.get("/", include_in_schema=False)
async def redirect():
    return RedirectResponse("/docs")


def get_prediction(image_data: bytes) -> bytes:

    # convert the image data to a numpy array
    image = np.array(Image.open(io.BytesIO(image_data)))

    prediction = model.predict(image)

    predicted_mask = ""

    # convert the processed image back to bytes
    is_success, buffer = cv2.imencode(".jpg", predicted_mask)
    if is_success:
        return buffer.tobytes()
    else:
        raise ValueError("Failed to encode image.")
    


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the image data
        image_data = await file.read()

        # predict the image
        predicted_mask = get_prediction(image_data)

        # Return the processed image
        return StreamingResponse(io.BytesIO(predicted_mask), media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8585)