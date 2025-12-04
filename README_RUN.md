Run instructions — Heart Disease project

Overview
--------
This document describes the minimal folders and files you must have in place before starting the FastAPI server that loads the ML model, and exact commands to start the app and the static frontend.

Important folders (ensure these exist and are reachable from the project root):

- `models/`
  - Required file: `models/best_model.joblib`
  - This is the serialized scikit-learn model loaded at startup by `src/app.py`.

- `data/`
  - Example dataset: `data/synthetic_heart_disease_dataset.csv` (used by frontend templates/downloads).
  - Not required for model startup, but useful for testing and sample downloads.

- `web/`
  - Contains the frontend static files (e.g. `menu.html`, `train.html`, `batch_check.html`, ...).
  - You can serve these files with a simple static server for local development.

- `src/`
  - Contains the FastAPI app: `src/app.py` (the API that loads the model and exposes `/`, `/predict`, `/predict/batch`).

Pre-requisites (recommended)
---------------------------
- Python 3.9+ (use same major/minor used to train the model if possible)
- Conda or virtualenv recommended
- The repo already contains `requirements.txt` with pinned libs (FastAPI, uvicorn, pandas, scikit-learn, joblib, ...)

Quick start (macOS / zsh)
-------------------------
Open a terminal and run these commands from the project root (`/Users/…/heart_disease`):

```bash
# 1) Activate environment
conda activate traffic_env   # or your env name

# 2) Install dependencies (one-time)
python -m pip install -r requirements.txt

# 3) Verify required folders/files
ls -l models/best_model.joblib
ls -l data/synthetic_heart_disease_dataset.csv
ls -l web/

# If any file is missing, copy or download them into the correct folder before proceeding.

# 4) Run the FastAPI app (recommended method)
# Use uvicorn to serve the ASGI app, pointing to the FastAPI app object in src.app:app
uvicorn src.app:app --host 127.0.0.1 --port 3000 --reload

# 5) (Optional) Serve the frontend static files on a different port
# From the project root (so '../data' and '../models' paths remain correct for frontend fetches)
python3 -m http.server 8090 --directory web

# Now open the frontend pages at:
# http://localhost:8090/menu.html  (or batch_check.html / train.html)
# The frontend will call the API at http://127.0.0.1:3000
```

Notes about `python3 app.py`
---------------------------
- The file `src/app.py` in this project defines a FastAPI app, but it does not include a builtin `if __name__ == "__main__"` Uvicorn runner. Running `python3 src/app.py` will import the module but not start a server.
- Recommended: use `uvicorn src.app:app` (shown above). If you must use `python3 app.py`, you can add a small runner file at project root (example below) that starts Uvicorn programmatically.

Example programmatic runner (optional)
-------------------------------------
Create a file `run_server.py` at project root with:

```python
# run_server.py
import uvicorn
if __name__ == '__main__':
    uvicorn.run('src.app:app', host='127.0.0.1', port=8000, reload=True)
```

Then run:
```bash
python3 run_server.py
# Server will start on http://127.0.0.1:3000
```

Health checks and test requests
-------------------------------
- Health check (browser or curl):

```
curl http://127.0.0.1:3000/
# Expected: {"status":"ok","message":"Heart Disease Prediction API is running"}
```

- Single prediction example (replace values with real test values):

```
curl -X POST http://127.0.0.1:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"Age":55,"Gender":"M","Weight":80,"Height":170,"BMI":27.7,"Smoking":"No","Alcohol_Intake":"Yes","Physical_Activity":"Low","Diet":"Mixed","Stress_Level":"Medium","Hypertension":1,"Diabetes":0,"Hyperlipidemia":1,"Family_History":0,"Previous_Heart_Attack":0,"Systolic_BP":135,"Diastolic_BP":85,"Heart_Rate":78,"Blood_Sugar_Fasting":95,"Cholesterol_Total":210}'
```

- Batch prediction example (send JSON list of dicts to `/predict/batch`), or use the frontend `batch_check.html` upload flow.

Troubleshooting
---------------
- Model not found error on startup: ensure `models/best_model.joblib` exists and is compatible with the installed scikit-learn version.
- Version mismatch: if model was trained with a different scikit-learn version, consider re-training or using the same scikit-learn version used for training.
- CORS errors from frontend: `src/app.py` already enables permissive CORS for development; tighten in production.

Production notes (optional)
---------------------------
- For production, run with a process manager (systemd, supervisor) and behind a reverse proxy (nginx).
- Use `--workers` with `gunicorn`+`uvicorn.workers.UvicornWorker` for concurrency or run Uvicorn with multiple workers via process manager.
- Keep `models/` and any secrets outside the webroot and set proper permissions.

If you want, tôi có thể:
- Add the small `run_server.py` runner to the repo for `python3 run_server.py` convenience.
- Add a short `Makefile` with targets `make run`, `make install`, `make serve-web`.
- Validate the installed model file and report back if any compatibility issues are detected.

----
(Generated by assistant)