import os
import io
import re
from collections import Counter
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import httpx
import numpy as np
from fastapi import FastAPI, BackgroundTasks, HTTPException
from paddleocr import PaddleOCR
from insightface.app import FaceAnalysis
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

app = FastAPI()

ocr_instance = None
face_app_instance = None

ENV_SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")


def append_query_params(url: str, params: dict) -> str:
    """
    Adiciona query params a uma URL sem quebrar se ela já tiver parâmetros.
    """
    parsed = urlparse(url)
    query = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query.update({k: v for k, v in params.items() if v is not None})
    new_query = urlencode(query)
    return urlunparse(parsed._replace(query=new_query))


def get_ocr():
    global ocr_instance
    if ocr_instance is None:
        print("Initializing PaddleOCR...")
        ocr_instance = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            show_log=False
        )
        print("PaddleOCR initialized.")
    return ocr_instance


def get_face_app():
    global face_app_instance
    if face_app_instance is None:
        print("Initializing InsightFace...")
        face_app_instance = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )
        face_app_instance.prepare(ctx_id=0, det_size=(640, 640))
        print("InsightFace initialized.")
    return face_app_instance


def download_image(url: str) -> np.ndarray:
    try:
        print(f"Downloading image from: {url}")
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()

        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
        print("Image downloaded successfully.")
        return np.array(img)

    except Exception as e:
        print(f"Error downloading image: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Erro ao baixar imagem: {str(e)}")


def post_callback(callback_url: str, payload: dict, service_role_key: str | None = None):
    token = service_role_key or ENV_SUPABASE_SERVICE_KEY

    if not token:
        raise Exception("Nenhuma service_role_key disponível para o callback.")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    print("=== CALLBACK DEBUG ===")
    print("POST URL:", callback_url)
    print("HEADERS:", {"Authorization": "Bearer ***", "Content-Type": "application/json"})
    print("PAYLOAD:", payload)
    print("TOKEN SOURCE:", "request" if service_role_key else "environment")

    try:
        with httpx.Client(timeout=60, follow_redirects=True) as client:
            response = client.post(callback_url, json=payload, headers=headers)

        print("CALLBACK STATUS:", response.status_code)
        print("CALLBACK RESPONSE:", response.text)
        response.raise_for_status()

    except Exception as e:
        print("CALLBACK ERROR:", str(e))
        raise Exception(f"Erro ao enviar callback: {str(e)}")


def safe_crop(img_np: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray | None:
    h, w = img_np.shape[:2]
    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = img_np[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def remove_top_and_bottom_noise(img_np: np.ndarray) -> np.ndarray:
    """
    Ignora áreas comuns de ruído:
    - topo: marca d'água/texto
    - rodapé: banner/logotipo da corrida
    """
    h, w = img_np.shape[:2]

    top_cut = int(h * 0.10)
    bottom_cut = int(h * 0.18)

    cropped = img_np[top_cut:h - bottom_cut, 0:w]
    print(f"Removed noisy bands | original_h={h} | top_cut={top_cut} | bottom_cut={bottom_cut} | new_shape={cropped.shape}")
    return cropped


def generate_ocr_variants(img_np: np.ndarray) -> list[np.ndarray]:
    pil = Image.fromarray(img_np)
    variants = []

    variants.append(np.array(pil))

    up2 = pil.resize((pil.width * 2, pil.height * 2), Image.Resampling.LANCZOS)
    variants.append(np.array(up2))

    gray = ImageOps.grayscale(up2).convert("RGB")
    variants.append(np.array(gray))

    high_contrast = ImageEnhance.Contrast(gray).enhance(2.2)
    variants.append(np.array(high_contrast))

    sharp = high_contrast.filter(ImageFilter.SHARPEN)
    variants.append(np.array(sharp))

    extra = ImageEnhance.Contrast(sharp).enhance(1.4).filter(ImageFilter.SHARPEN)
    variants.append(np.array(extra))

    return variants


def normalize_ocr_text(text: str) -> str:
    return re.sub(r"[^0-9]", "", text or "")


def is_plausible_bib_number(number: int) -> bool:
    return 1 <= number <= 9999


def score_number(number: int, confidence: float, source: str) -> float:
    score = confidence

    digit_len = len(str(number))
    if digit_len == 3:
        score += 0.30
    elif digit_len == 2:
        score += 0.20
    elif digit_len == 4:
        score += 0.05

    if source == "torso":
        score += 0.40
    elif source == "global":
        score += 0.05

    if 1900 <= number <= 2100:
        score -= 0.80

    return score


def run_ocr_on_region(img_np: np.ndarray, source: str) -> list[tuple[int, float]]:
    ocr = get_ocr()
    variants = generate_ocr_variants(img_np)

    found = []

    for idx, variant in enumerate(variants):
        print(f"Running OCR variant {idx + 1}/{len(variants)} | source={source}")
        result = ocr.ocr(variant, cls=True)

        if not result or not result[0]:
            continue

        for line in result[0]:
            try:
                text = line[1][0]
                confidence = float(line[1][1])
            except Exception:
                continue

            if confidence < 0.55:
                continue

            digits = normalize_ocr_text(text)
            if not digits:
                continue

            if len(digits) > 4:
                continue

            try:
                number = int(digits)
            except ValueError:
                continue

            if not is_plausible_bib_number(number):
                continue

            s = score_number(number, confidence, source)
            found.append((number, s))
            print(f"OCR candidate | source={source} | raw='{text}' | digits={digits} | conf={confidence:.3f} | score={s:.3f}")

    return found


def extract_torso_regions(img_np: np.ndarray) -> list[np.ndarray]:
    face_app = get_face_app()
    faces = face_app.get(img_np)

    regions = []

    if not faces:
        print("No faces found for torso estimation.")
        return regions

    print(f"Faces found for torso estimation: {len(faces)}")

    for i, face in enumerate(faces):
        try:
            x1, y1, x2, y2 = [int(v) for v in face.bbox.tolist()]
        except Exception:
            continue

        fw = max(1, x2 - x1)
        fh = max(1, y2 - y1)
        cx = x1 + fw // 2

        torso_x1 = int(cx - fw * 1.6)
        torso_x2 = int(cx + fw * 1.6)
        torso_y1 = int(y2 + fh * 0.4)
        torso_y2 = int(y2 + fh * 4.8)

        crop = safe_crop(img_np, torso_x1, torso_y1, torso_x2, torso_y2)

        if crop is None:
            continue

        ch, cw = crop.shape[:2]
        if cw < 50 or ch < 50:
            continue

        print(
            f"Torso region {i + 1} | face=({x1},{y1},{x2},{y2}) "
            f"| torso=({torso_x1},{torso_y1},{torso_x2},{torso_y2}) "
            f"| crop_shape={crop.shape}"
        )
        regions.append(crop)

    return regions


def select_best_bib_numbers(global_candidates: list[tuple[int, float]], torso_candidates: list[tuple[int, float]]) -> list[int]:
    weighted_scores = Counter()
    occurrences = Counter()

    for number, score in global_candidates + torso_candidates:
        weighted_scores[number] += score
        occurrences[number] += 1

    if not weighted_scores:
        return []

    ranking = sorted(
        weighted_scores.keys(),
        key=lambda n: (occurrences[n], weighted_scores[n]),
        reverse=True
    )

    print("BIB ranking debug:")
    for n in ranking:
        print(f"  number={n} | occurrences={occurrences[n]} | weighted_score={weighted_scores[n]:.3f}")

    return ranking[:6]


def extract_bib_numbers(img_np: np.ndarray) -> list[int]:
    cleaned = remove_top_and_bottom_noise(img_np)

    print("Running OCR on cleaned global image...")
    global_candidates = run_ocr_on_region(cleaned, source="global")

    print("Estimating torso regions...")
    torso_regions = extract_torso_regions(cleaned)

    torso_candidates = []
    for idx, region in enumerate(torso_regions):
        print(f"Running torso OCR for region {idx + 1}/{len(torso_regions)}")
        torso_candidates.extend(run_ocr_on_region(region, source="torso"))

    final_numbers = select_best_bib_numbers(global_candidates, torso_candidates)

    if len(final_numbers) > 1:
        filtered = [n for n in final_numbers if not (1900 <= n <= 2100)]
        if filtered:
            final_numbers = filtered

    print("FINAL BIB NUMBERS:", final_numbers)
    return final_numbers


async def process_photo_task(photo_id: str, image_url: str, callback_url: str, service_role_key: str | None = None):
    try:
        print(f"Starting process_photo_task | photo_id={photo_id}")
        img = download_image(image_url)

        face_app = get_face_app()

        print("Running race bib extraction...")
        numbers = extract_bib_numbers(img)

        if numbers:
            print("BIB NUMBERS DETECTED:", numbers)
            jersey_callback_url = append_query_params(callback_url, {"action": "jersey_numbers"})
            post_callback(
                jersey_callback_url,
                {
                    "photo_id": photo_id,
                    "jersey_numbers": numbers,
                },
                service_role_key=service_role_key,
            )
        else:
            print("NO BIB NUMBERS DETECTED")

        print("Running face detection...")
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

            print(f"FACES DETECTED: {len(face_data)}")

            faces_callback_url = append_query_params(callback_url, {"action": "face_embeddings"})
            post_callback(
                faces_callback_url,
                {
                    "photo_id": photo_id,
                    "faces": face_data,
                },
                service_role_key=service_role_key,
            )
        else:
            print("NO FACES DETECTED")

        print(f"Finished process_photo_task | photo_id={photo_id}")

    except Exception as e:
        print(f"ERROR in process_photo_task | photo_id={photo_id} | error={str(e)}")


@app.post("/process")
async def process(
    photo_id: str,
    image_url: str,
    callback_url: str,
    bg: BackgroundTasks,
    service_role_key: str | None = None
):
    print("=== /process REQUEST RECEIVED ===")
    print("photo_id:", photo_id)
    print("image_url:", image_url)
    print("callback_url:", callback_url)
    print("service_role_key received:", bool(service_role_key))

    bg.add_task(process_photo_task, photo_id, image_url, callback_url, service_role_key)
    return {"status": "queued"}


@app.post("/process-selfie")
async def process_selfie(
    user_id: str,
    image_url: str,
    callback_url: str,
    service_role_key: str | None = None
):
    try:
        print("=== /process-selfie REQUEST RECEIVED ===")
        print("user_id:", user_id)
        print("image_url:", image_url)
        print("callback_url:", callback_url)
        print("service_role_key received:", bool(service_role_key))

        img = download_image(image_url)

        face_app = get_face_app()
        faces = face_app.get(img)

        if not faces:
            print("NO FACE DETECTED FOR SELFIE")
            return {"error": "Nenhum rosto detectado"}

        embedding = faces[0].embedding.tolist()
        print("SELFIE FACE DETECTED")

        # IMPORTANTE:
        # Para o fluxo anônimo de selfie, usa o callback_url exatamente como veio,
        # sem forçar ?action=user_embedding.
        post_callback(
            callback_url,
            {
                "user_id": user_id,
                "embedding": embedding,
            },
            service_role_key=service_role_key,
        )

        return {"status": "ok"}

    except Exception as e:
        print(f"ERROR in process_selfie | user_id={user_id} | error={str(e)}")
        raise HTTPException(status_code=500, detail=f"Erro no processamento da selfie: {str(e)}")


@app.get("/health")
async def health():
    return {"status": "ok"}
