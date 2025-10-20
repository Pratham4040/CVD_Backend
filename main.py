import io
import os
import logging
from typing import List

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# Color blindness transformation matrices
CB_MATRICES = {
    "protanopia": np.array([[0.567, 0.433, 0.0],
                           [0.558, 0.442, 0.0],
                           [0.0,   0.242, 0.758]]),
    "deuteranopia": np.array([[0.625, 0.375, 0.0],
                             [0.70,  0.30,  0.0],
                             [0.0,   0.30,  0.70]]),
    "tritanopia": np.array([[0.95,  0.05,  0.0],
                           [0.0,   0.433, 0.567],
                           [0.0,   0.475, 0.525]])
}

def simulate_cvd(img_rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Simulate color vision deficiency using transformation matrix."""
    img_norm = img_rgb.astype(np.float32) / 255.0
    transformed = img_norm @ matrix.T
    transformed = np.clip(transformed, 0.0, 1.0) * 255
    return transformed.astype(np.uint8)

############################
# Configuration
############################

def _get_list_env(name: str, default: str) -> List[str]:
    raw = os.getenv(name, default)
    return [v.strip() for v in raw.split(",") if v.strip()]

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
ALLOWED_ORIGINS = _get_list_env(
    "CORS_ALLOW_ORIGINS",
    "https://cvdsimulator.netlify.app/,http://localhost:5173,http://127.0.0.1:5173,http://localhost:5174,http://127.0.0.1:5174",
)
ALLOWED_HOSTS = _get_list_env("ALLOWED_HOSTS", "*")
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "1"))  # maximum upload size in MB
MAX_IMAGE_DIM = int(os.getenv("MAX_IMAGE_DIM", "4096"))  # maximum width/height; larger images will be scaled down

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("cvd-simulator")

# Create FastAPI app instance
app = FastAPI(
    title="CVD Simulator API",
    description="Color Vision Deficiency Simulator Backend",
    version=os.getenv("APP_VERSION", "1.0.0"),
)

# Trusted hosts (protects against Host header attacks)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

# Enable CORS to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add basic security headers to responses."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=()")
        return response


app.add_middleware(SecurityHeadersMiddleware)

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {"message": "CVD Simulator API is running"}


@app.get("/healthz")
async def healthz():
    """Kubernetes-style liveness/readiness check."""
    return {"status": "ok", "version": app.version}


@app.get("/api/version")
async def version():
    return {"version": app.version}

@app.post("/api/simulate")
async def simulate_cvd_endpoint(
    file: UploadFile = File(...),
    cvd_type: str = Form(...)
):
    """
    Simulate color vision deficiency on uploaded image.
    
    Args:
        file: Uploaded image file
        cvd_type: Type of CVD ('protanopia', 'deuteranopia', or 'tritanopia')
    
    Returns:
        StreamingResponse containing the processed image
    """
    try:
        # Validate CVD type
        if cvd_type not in CB_MATRICES:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid CVD type. Must be one of: {list(CB_MATRICES.keys())}"
            )
        
        # Validate content type
        if not (file.content_type and file.content_type.startswith("image/")):
            raise HTTPException(status_code=400, detail="Only image uploads are supported")

        # Read uploaded file bytes
        file_bytes = await file.read()

        # Enforce max upload size
        max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
        if len(file_bytes) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB} MB")
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_bytes, np.uint8)
        
        # Decode image from bytes (OpenCV reads in BGR format)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Downscale very large images to protect memory/CPU
        h, w = img_bgr.shape[:2]
        if max(h, w) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Convert BGR to RGB for processing
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Apply CVD simulation
        matrix = CB_MATRICES[cvd_type]
        simulated_rgb = simulate_cvd(img_rgb, matrix)
        
        # Convert back to BGR for encoding
        simulated_bgr = cv2.cvtColor(simulated_rgb, cv2.COLOR_RGB2BGR)

        # Encode image as JPEG (quality configurable via env)
        jpg_quality = int(os.getenv("JPEG_QUALITY", "92"))
        success, encoded_img = cv2.imencode(
            '.jpg', simulated_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to encode processed image")

        # Convert to bytes
        img_bytes = encoded_img.tobytes()

        # Return as streaming response
        response = StreamingResponse(
            io.BytesIO(img_bytes),
            media_type="image/jpeg",
            headers={"Content-Disposition": f"inline; filename=cvd_{cvd_type}.jpg"}
        )
        return response
        
    except Exception as e:
        logger.exception("Error processing image")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Processing error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)