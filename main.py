import os
import io
import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis
from PIL import Image

app = FastAPI()

ocr_instance = None
face_app_instance = None

SUPABASE_SERVICE_KEY = os.environ["SUPABASE_SERVICE_KEY"]


def get_ocr():
    global ocr_instance
    if ocr_instance is None:
        ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False
        )
    return ocr_instance


def get_face_app():
    global face_app_instance
    if face_app_instance is None:
        face_app_instance = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        face_app_instance.prepare(ctx_id=0, det_size=(640, 640))
    return face_app_instance


def download_image(url: str) -> np.ndarray:
    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        return np.array(img)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro ao baixar imagem: {str(e)}")


def post_callback(callback_url: str, payload: dict):
    headers = {
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.post(callback_url, json=payload, headers=headers)
            response.raise_for_status()
    except Exception as e:
        raise Exception(f"Erro ao enviar callback: {str(e)}")


async def process_photo_task(photo_id: str, image_url: str, callback_url: str):
    try:
        img = download_image(image_url)

        ocr = get_ocr()
        face_app = get_face_app()

        result = ocr.ocr(img, cls=True)
        numbers = []

        if result and result[0]:
            for line in result[0]:
                text = line[1][0]
                digits = "".join(c for c in text if c.isdigit())
                if digits:
                    try:
                        number = int(digits)
                        if 1 <= number <= 9999:
                            numbers.append(number)
                    except ValueError:
                        pass

        if numbers:
            post_callback(
                f"{callback_url}?action=jersey_numbers",
                {
                    "photo_id": photo_id,
                    "jersey_numbers": list(set(numbers)),
                },
            )

        faces = face_app.get(img)
        if faces:
            face_data = []

            for face in faces:
                bbox = face.bbox.tolist()
                face_data.append(
                    {
                        "embedding": face.embedding.tolist(),
                        "bbox": {
                            "x": bbox[0],
                            "y": bbox[1],
                            "w": bbox[2] - bbox[0],
                            "h": bbox[3] - bbox[1],
                        },
                    }
                )

            post_callback(
                f"{callback_url}?action=face_embeddings",
                {
                    "photo_id": photo_id,
                    "faces": face_data,
                },
            )

    except Exception as e:
        print(f"Erro em process_photo_task: {str(e)}")


@app.post("/process")
async def process(photo_id: str, image_url: str, callback_url: str, bg: BackgroundTasks):
    bg.add_task(process_photo_task, photo_id, image_url, callback_url)
    return {"status": "queued"}


@app.post("/process-selfie")
async def process_selfie(user_id: str, image_url: str, callback_url: str):
    try:
        img = download_image(image_url)

        face_app = get_face_app()
        faces = face_app.get(img)

        if not faces:
            return {"error": "Nenhum rosto detectado"}

        embedding = faces[0].embedding.tolist()

        post_callback(
            f"{callback_url}?action=user_embedding",
            {
                "user_id": user_id,
                "embedding": embedding,
            },
        )

        return {"status": "ok"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento da selfie: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
