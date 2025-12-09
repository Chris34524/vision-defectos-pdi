# src/api/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from ultralytics import YOLO
from PIL import Image
from fastapi.staticfiles import StaticFiles
import io

app = FastAPI(
    title="Vision de Defectos - MVP",
    version="0.1.0",
)

# ⚠️ Por ahora usaremos el modelo genérico YOLOv8n (COCO)
# En tu repo ya tienes yolov8n.pt en la raíz, luego lo moveremos a models/
MODEL_PATH = "yolov8n.pt"
model = YOLO(MODEL_PATH)


@app.get("/health")
def health():
    """Verifica que el servicio está vivo y el modelo cargado."""
    return {
        "status": "ok",
        "model_loaded": MODEL_PATH,
    }


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    """
    Recibe una imagen, corre YOLO y devuelve detecciones en JSON.
    """
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Inferencia con YOLO
        results = model.predict(img, imgsz=640, conf=0.25)[0]

        detections = []
        for box in results.boxes:
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            score = float(box.conf[0].item())
            class_name = results.names[cls_id]

            detections.append({
                "box": {
                    "x1": xyxy[0],
                    "y1": xyxy[1],
                    "x2": xyxy[2],
                    "y2": xyxy[3],
                },
                "class_id": cls_id,
                "class_name": class_name,
                "score": score,
            })

        return JSONResponse(content={
            "num_detections": len(detections),
            "detections": detections,
        })

    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
        )
app.mount("/", StaticFiles(directory="static", html=True), name="static")