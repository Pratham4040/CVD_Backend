import io
import os
import logging
from typing import List

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi import Body
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

############################
# Color utilities
############################

def _rgb_to_hex(rgb: np.ndarray) -> str:
    r, g, b = [int(max(0, min(255, v))) for v in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"

def _hex_to_rgb(hex_str: str) -> np.ndarray:
    h = hex_str.lstrip('#')
    if len(h) == 3:
        h = ''.join([c*2 for c in h])
    return np.array([int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)], dtype=np.float32)

def _srgb_to_linear(c: np.ndarray) -> np.ndarray:
    c = c / 255.0
    return np.where(c <= 0.04045, c/12.92, ((c + 0.055)/1.055) ** 2.4)

def _relative_luminance(rgb: np.ndarray) -> float:
    # rgb expected 0..255
    r, g, b = _srgb_to_linear(rgb)
    return float(0.2126*r + 0.7152*g + 0.0722*b)

def _contrast_ratio(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    L1 = _relative_luminance(rgb1)
    L2 = _relative_luminance(rgb2)
    L_light = max(L1, L2)
    L_dark = min(L1, L2)
    return (L_light + 0.05) / (L_dark + 0.05)

def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    # expects 0..255
    arr = np.uint8(np.clip(rgb, 0, 255)).reshape(1, 1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).reshape(3)
    return lab.astype(np.float32)

def _delta_e(rgb1: np.ndarray, rgb2: np.ndarray) -> float:
    lab1 = _rgb_to_lab(rgb1)
    lab2 = _rgb_to_lab(rgb2)
    return float(np.linalg.norm(lab1 - lab2))

def _simulate_color_cvd(rgb: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    v = (rgb.astype(np.float32) / 255.0) @ matrix.T
    v = np.clip(v, 0.0, 1.0) * 255.0
    return v.astype(np.float32)

def _rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    arr = np.uint8(np.clip(rgb, 0, 255)).reshape(1, 1, 3)
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).reshape(3).astype(np.float32)
    return hsv

def _hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    arr = np.float32(hsv).reshape(1, 1, 3)
    arr = np.uint8(np.clip(arr, 0, 255))
    rgb = cv2.cvtColor(arr, cv2.COLOR_HSV2RGB).reshape(3).astype(np.float32)
    return rgb

def _ensure_contrast(base_rgb: np.ndarray, against_rgb: np.ndarray, min_ratio: float = 4.5) -> np.ndarray:
    # Adjust V (value) to achieve contrast threshold
    rgb = base_rgb.copy()
    tries = 0
    while _contrast_ratio(rgb, against_rgb) < min_ratio and tries < 20:
        hsv = _rgb_to_hsv(rgb)
        if _relative_luminance(rgb) <= _relative_luminance(against_rgb):
            hsv[2] = max(0, hsv[2] - 8)  # darken
        else:
            hsv[2] = min(255, hsv[2] + 8)  # brighten
        rgb = _hsv_to_rgb(hsv)
        tries += 1
    return rgb

def _extract_palette(img_rgb: np.ndarray, k: int = 5) -> List[dict]:
    # Flatten pixels
    pixels = img_rgb.reshape(-1, 3).astype(np.float32)
    # sample up to 100k pixels for speed
    if pixels.shape[0] > 100000:
        idx = np.random.choice(pixels.shape[0], 100000, replace=False)
        pixels = pixels[idx]
    # K-means
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    attempts = 3
    flags = cv2.KMEANS_PP_CENTERS
    compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, attempts, flags)
    centers = centers.astype(np.float32)
    counts = np.bincount(labels.flatten(), minlength=k).astype(np.float32)
    total = float(counts.sum() + 1e-9)
    # sort by count desc
    order = np.argsort(-counts)
    palette = []
    for i in order:
        rgb = centers[i]
        pct = float(counts[i] / total)
        palette.append({
            "rgb": [float(v) for v in rgb],
            "hex": _rgb_to_hex(rgb),
            "percent": pct,
        })
    return palette

def _cvd_confusion_scores(colors: List[np.ndarray]) -> List[dict]:
    results = []
    n = len(colors)
    for i in range(n):
        for j in range(i+1, n):
            c1 = colors[i]
            c2 = colors[j]
            # Original contrast
            orig_contrast = _contrast_ratio(c1, c2)
            
            min_de = 1e9
            worst_cvd = None
            min_cvd_contrast = 1e9
            for cvd_name, m in CB_MATRICES.items():
                s1 = _simulate_color_cvd(c1, m)
                s2 = _simulate_color_cvd(c2, m)
                de = _delta_e(s1, s2)
                cvd_contrast = _contrast_ratio(s1, s2)
                if de < min_de:
                    min_de = de
                    worst_cvd = cvd_name
                if cvd_contrast < min_cvd_contrast:
                    min_cvd_contrast = cvd_contrast
            
            # risk 1 at de<=10, 0 at de>=40
            risk = max(0.0, min(1.0, (40.0 - min_de) / 30.0))
            
            # Also flag if contrast drops below 3.0 under CVD
            contrast_loss = orig_contrast - min_cvd_contrast
            
            results.append({
                "c1": _rgb_to_hex(c1),
                "c2": _rgb_to_hex(c2),
                "minDeltaE": float(min_de),
                "risk": float(risk),
                "worstCVD": worst_cvd,
                "originalContrast": round(float(orig_contrast), 2),
                "cvdContrast": round(float(min_cvd_contrast), 2),
                "contrastLoss": round(float(contrast_loss), 2),
                "readable": min_cvd_contrast >= 3.0,  # minimum for large text
            })
    return results

def _suggest_cvd_safe_variant(color_rgb: np.ndarray, others: List[np.ndarray]) -> np.ndarray:
    # Try small hue shifts to maximize min DeltaE against others under CVD sims
    best_rgb = color_rgb.copy()
    best_score = -1.0
    hsv0 = _rgb_to_hsv(color_rgb)
    for dh in range(-30, 31, 5):
        hsv = hsv0.copy()
        hsv[0] = (hsv[0] + (dh % 180)) % 180  # OpenCV HSV hue range 0..179
        hsv[1] = np.clip(hsv[1] * 1.05, 0, 255)
        candidate = _hsv_to_rgb(hsv)
        # score: min DeltaE vs others under all CVD, plus AA against white/black
        min_de = 1e9
        for o in others:
            for m in CB_MATRICES.values():
                s1 = _simulate_color_cvd(candidate, m)
                s2 = _simulate_color_cvd(o, m)
                de = _delta_e(s1, s2)
                min_de = min(min_de, de)
        # encourage AA contrast against white and black
        cr_white = _contrast_ratio(candidate, np.array([255, 255, 255], dtype=np.float32))
        cr_black = _contrast_ratio(candidate, np.array([0, 0, 0], dtype=np.float32))
        score = min_de + (5.0 if cr_white >= 4.5 else 0.0) + (5.0 if cr_black >= 4.5 else 0.0)
        if score > best_score:
            best_score = score
            best_rgb = candidate
    return best_rgb

def _build_tokens_exports(palette_hex: List[str]) -> dict:
    css_lines = [":root {"]
    tw_lines = ["// Tailwind snippet", "// Add into tailwind.config theme.extend.colors", "colors: {"]
    mapping = {}
    for idx, hx in enumerate(palette_hex, start=1):
        name = f"brand{idx}"
        mapping[name] = hx
        css_lines.append(f"  --{name}: {hx};")
        tw_lines.append(f"  {name}: 'var(--{name})',")
    css_lines.append("}")
    tw_lines.append("}")
    css_content = "\n".join(css_lines) + "\n"
    tw_snippet = "\n".join(tw_lines) + "\n"
    # Unified diff to add frontend/src/tokens.css
    diff = (
        "*** Begin Patch\n"
        "*** Add File: frontend/src/tokens.css\n" + css_content + "\n*** End Patch"
    )
    return {
        "cssVariables": css_content,
        "tailwindSnippet": tw_snippet,
        "diff": diff,
        "mapping": mapping,
    }

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


@app.post("/api/palette")
async def extract_palette(
    file: UploadFile = File(...),
    k: int = Form(5),
):
    """Extract dominant colors from image and return palette with percentages."""
    try:
        if not (file.content_type and file.content_type.startswith("image/")):
            raise HTTPException(status_code=400, detail="Only image uploads are supported")
        data = await file.read()
        max_bytes = int(MAX_UPLOAD_MB * 1024 * 1024)
        if len(data) > max_bytes:
            raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_MB} MB")
        nparr = np.frombuffer(data, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        h, w = img_bgr.shape[:2]
        if max(h, w) > MAX_IMAGE_DIM:
            scale = MAX_IMAGE_DIM / float(max(h, w))
            img_bgr = cv2.resize(img_bgr, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        k = int(max(2, min(10, k)))
        palette = _extract_palette(img_rgb, k=k)
        return {"palette": palette}
    except Exception as e:
        logger.exception("palette extraction failed")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Palette error")


@app.post("/api/palette/analyze")
async def analyze_palette(
    palette: List[str] = Body(..., embed=True, description="List of hex colors"),
    pairs: List[dict] = Body(default=None, embed=True, description="Optional list of {fg,bg} hex pairs for contrast checks"),
):
    """Compute WCAG contrast and CVD confusion risk; suggest safer variants."""
    try:
        colors_rgb = [ _hex_to_rgb(hx) for hx in palette ]
        # WCAG for provided pairs (if any)
        wcag = []
        if pairs:
            for p in pairs:
                fg = _hex_to_rgb(p.get("fg"))
                bg = _hex_to_rgb(p.get("bg"))
                cr = _contrast_ratio(fg, bg)
                wcag.append({
                    "fg": p.get("fg"),
                    "bg": p.get("bg"),
                    "contrast": round(cr, 2),
                    "passAA": cr >= 4.5,
                    "passAAA": cr >= 7.0,
                })
        # CVD confusion across palette
        confusions = _cvd_confusion_scores(colors_rgb)
        # --- CVD visibility scoring ---
        # For each CVD type, count pairs with cvdContrast >= 3.0 and DeltaE >= 20
        cvd_types = ["protanopia", "deuteranopia", "tritanopia"]
        cvd_scores = {}
        for cvd in cvd_types:
            total = 0
            good = 0
            for c in confusions:
                if c["worstCVD"] == cvd:
                    total += 1
                    if c["cvdContrast"] >= 3.0 and c["minDeltaE"] >= 20:
                        good += 1
            score = (good / total) if total else 1.0
            cvd_scores[cvd] = round(score * 100, 1)
        # Overall: mean of the three
        overall = sum(cvd_scores.values()) / len(cvd_scores) if cvd_scores else 100.0
        cvd_scores["overall"] = round(overall, 1)
        # Suggestions
        suggestions = []
        # For high-risk pairs, suggest variant on first color
        for c in confusions:
            if c["risk"] >= 0.5:
                c1 = _hex_to_rgb(c["c1"])
                others = [ _hex_to_rgb(c["c2"]) ]
                new_rgb = _suggest_cvd_safe_variant(c1, others)
                suggestions.append({
                    "type": "cvd",
                    "target": c["c1"],
                    "against": c["c2"],
                    "suggested": _rgb_to_hex(new_rgb),
                    "deltaEGain": round(float(_delta_e(_simulate_color_cvd(new_rgb, list(CB_MATRICES.values())[0]), _simulate_color_cvd(others[0], list(CB_MATRICES.values())[0])) - c["minDeltaE"]), 2),
                })
        # For contrast failures, suggest switching to black/white or adjust V
        if wcag:
            for r in wcag:
                if not r["passAA"]:
                    fg = _hex_to_rgb(r["fg"])
                    bg = _hex_to_rgb(r["bg"])
                    options = [
                        np.array([0,0,0], dtype=np.float32),
                        np.array([255,255,255], dtype=np.float32),
                    ]
                    # Try adjusted fg first
                    adj_fg = _ensure_contrast(fg, bg, 4.5)
                    options.append(adj_fg)
                    best = None
                    best_cr = -1
                    for opt in options:
                        cr = _contrast_ratio(opt, bg)
                        if cr > best_cr:
                            best_cr = cr
                            best = opt
                    suggestions.append({
                        "type": "contrast",
                        "pair": {"fg": r["fg"], "bg": r["bg"]},
                        "suggestedFg": _rgb_to_hex(best),
                        "contrast": round(float(best_cr), 2),
                        "delta": round(float(best_cr - r["contrast"]), 2),
                    })
        return {"wcag": wcag, "cvd": confusions, "cvdScores": cvd_scores, "suggestions": suggestions}
    except Exception as e:
        logger.exception("palette analyze failed")
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail="Analyze error")


@app.post("/api/palette/export")
async def export_tokens(
    palette: List[str] = Body(..., embed=True, description="List of hex colors"),
):
    """Return CSS variables, Tailwind snippet, and a unified-diff to add a tokens.css file."""
    try:
        exports = _build_tokens_exports(palette)
        return exports
    except Exception as e:
        logger.exception("export tokens failed")
        raise HTTPException(status_code=500, detail="Export error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)