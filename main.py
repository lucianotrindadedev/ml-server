import os, io, httpx, numpy as np
from fastapi import FastAPI, BackgroundTasks
from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis
from PIL import Image

app = FastAPI()

ocr_instance = None
face_app_instance = None

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
PROCESS_URL = f"{SUPABASE_URL}/functions/v1/process-results"

headers = {
    "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json"
}

def get_ocr():
    global ocr_instance
    if ocr_instance is None:
        ocr_instance = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    return ocr_instance

def get_face_app():
    global face_app_instance
    if face_app_instance is None:
        face_app_instance = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app_instance.prepare(ctx_id=0, det_size=(640, 640))
    return face_app_instance

def download_image(url: str) -> np.ndarray:
    resp = httpx.get(url, timeout=30)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return np.array(img)

async def process_photo(photo_id: str, image_url: str):
    img = download_image(image_url)

    ocr = get_ocr()
    face_app = get_face_app()

    result = ocr.ocr(img, cls=True)
    numbers = []
    for line in (result[0] or []):
        text = line[1][0]
        digits = ''.join(c for c in text if c.isdigit())
        if digits and 1 <= int(digits) <= 9999:
            numbers.append(int(digits))

    if numbers:
        httpx.post(
            f"{PROCESS_URL}?action=jersey_numbers",
            json={"photo_id": photo_id, "jersey_numbers": list(set(numbers))},
            headers=headers,
            timeout=15
        )

    faces = face_app.get(img)
    if faces:
        face_data = []
        for face in faces:
            bbox = face.bbox.tolist()
            face_data.append({
                "embedding": face.embedding.tolist(),
                "bbox": {
                    "x": bbox[0],
                    "y": bbox[1],
                    "w": bbox[2] - bbox[0],
                    "h": bbox[3] - bbox[1]
                }
            })

        httpx.post(
            f"{PROCESS_URL}?action=face_embeddings",
            json={"photo_id": photo_id, "faces": face_data},
            headers=headers,
            timeout=15
        )

@app.post("/process")
async def process(photo_id: str, image_url: str, bg: BackgroundTasks):
    bg.add_task(process_photo, photo_id, image_url)
    return {"status": "queued"}

@app.post("/process-selfie")
async def process_selfie(user_id: str, image_url: str):
    img = download_image(image_url)

    face_app = get_face_app()
    faces = face_app.get(img)

    if not faces:
        return {"error": "Nenhum rosto detectado"}

    embedding = faces[0].embedding.tolist()

    httpx.post(
        f"{PROCESS_URL}?action=user_embedding",
        json={"user_id": user_id, "embedding": embedding},
        headers=headers,
        timeout=15
    )

    return {"status": "ok"}

@app.get("/health")
async def health():
    return {"status": "ok"}
