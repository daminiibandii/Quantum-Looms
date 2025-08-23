# Quantum Assignment Starter (Backend + Frontend)

This is a tiny, beginner-friendly starter to connect your frontend website to a Python backend that
runs **classical** and **quantum-like (simulated annealing)** assignment optimization.

## What you get
- **FastAPI backend** at `http://127.0.0.1:8000`
- **Endpoint**: `POST /optimize` — send a cost matrix, get assignments back
- **Frontend**: static `index.html` that calls the API and displays results

---

## Step-by-step (Windows-friendly)

1) **Install Python 3.10+** from python.org and check in Terminal:
```bash
python --version
```

2) **Open a terminal** (Command Prompt or PowerShell) and go to the project folder:
```bash
cd quantum-assignment-starter
```

3) **Create & activate a virtual environment**:
```bash
python -m venv .venv
.venv\Scripts\activate
```

4) **Install dependencies**:
```bash
pip install -r requirements.txt
```

5) **Run the backend** (FastAPI + Uvicorn):
```bash
uvicorn backend.app:app --reload
```
You should see: `Uvicorn running on http://127.0.0.1:8000`

6) **Open the frontend** by double-clicking `frontend/index.html` (or open it in your browser).
   - Make sure the backend from step 5 is still running.

7) **Try it**: Click **Optimize**. You’ll see **Classical** vs **Quantum (local SA)** results.

---

## How to change inputs
- Edit the **Cost Matrix** textarea (rows = orders, columns = weavers).
- Penalty controls the strength of the QUBO constraint.
- Solver dropdown lets you compare Classical vs Quantum(Local).

---

## Optional: Real quantum hardware
This starter uses local **simulated annealing** (no cloud needed). If you have a D-Wave account,
you can extend `backend/app.py` to use a real QPU. For a 24h demo, local SA is perfect.

---

## Troubleshooting
- If the browser can’t reach the API, make sure Uvicorn is running and the URL is `http://127.0.0.1:8000`.
- If `ortools` install fails, run `pip install --upgrade pip` then re-install.
- If CORS blocks requests, we already allow all origins in `app.py` via `CORSMiddleware`.
- If you edit ports or host, update the fetch URL in `frontend/index.html`.

---

## File structure
```
quantum-assignment-starter/
├─ backend/
│  └─ app.py
├─ frontend/
│  └─ index.html
├─ requirements.txt
└─ README.md
```
