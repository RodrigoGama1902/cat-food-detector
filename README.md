# Cat Bowl Server

Flask API that detects how full a cat's food bowl is from an image. Supports day and night (IR) profiles, ROI calibration, and a live configuration UI.

---

## Requirements

- Docker & Docker Compose **or** Python 3.10+

---

## Running with Docker (recommended)

```bash
docker compose up --build
```

The server listens on **port 8000**. Configuration is persisted in `./data/config.json` via a volume mount.

---

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

> The server will start on `http://localhost:5000` by default (Flask dev mode).

To run with gunicorn (production-like):

```bash
gunicorn -w 1 -b 0.0.0.0:8000 app:app
```

---

## Configuration UI

Open `http://localhost:8000/` in a browser to access the calibration and ROI configuration interface.

---

## Endpoints

### `POST /detect`

Run food detection on an image. Returns whether food is present and how full the bowl is.

**Request** – send the image as:
- a multipart form field named `image`, **or**
- raw bytes in the request body

**Query parameters** (all optional — defaults come from the saved config):

| Parameter | Type | Description |
|---|---|---|
| `night_mode` | bool | Force night (`true`) or day (`false`) profile. Auto-detected from image saturation if omitted. |
| `threshold` | int | Binarization threshold (0–255). |
| `minimum_coverage` | float | Minimum ratio to consider food present (0–1). |
| `full_coverage` | float | Ratio considered "full" (0–1). |
| `method` | string | Detection method: `texture`, `brightness`, or `cluster`. |

**Response `200`**

```json
{
  "food_present": true,
  "coverage": 0.62,
  "raw_coverage": 0.18,
  "night_mode": false,
  "auto_detected": true,
  "mean_saturation": 45.3
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `food_present` | bool | `true` se `raw_coverage >= minimum_coverage`. |
| `coverage` | float | Cobertura normalizada (0–1) usando os limites empty/full calibrados. |
| `raw_coverage` | float | Fração bruta de pixels identificados como comida (0–1), antes da normalização. |
| `night_mode` | bool | Perfil usado na detecção (`true` = noite, `false` = dia). |
| `auto_detected` | bool | `true` se o perfil foi inferido automaticamente pela saturação da imagem. |
| `mean_saturation` | float | Saturação média da imagem (HSV), usada para a auto-detecção dia/noite. |

**Example with curl:**

```bash
curl -X POST http://localhost:8000/detect \
     -F "image=@photo.jpg"
```

---

### `GET /status`

Returns the **last** detection result cached from the most recent `POST /detect` call. Useful for polling without sending images repeatedly.

**Response `200`**

```json
{
  "food_present": true,
  "coverage": 0.62,
  "latency_ms": 34,
  "last_detection": "2026-06-28T12:00:00+00:00"
}
```

**Response `503`** – if `/detect` has never been called:

```json
{ "error": "no detection yet" }
```

**Example with curl:**

```bash
curl http://localhost:8000/status
```

---

## Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns `{"status": "ok"}`. |
| `GET` | `/api/config` | Returns the current saved configuration. |
| `POST` | `/api/config` | Persists a full config (ROI, profiles, etc.). |
| `POST` | `/api/roi` | Persists only the ROI `[x, y, w, h]`. |
| `POST` | `/api/calibrate/upload` | Upload empty/medium/full images for a profile. |
| `GET` | `/api/calibrate/preview` | Live detection preview for a profile's params. |
| `POST` | `/api/calibrate/save` | Persist a profile's calibrated parameters. |
