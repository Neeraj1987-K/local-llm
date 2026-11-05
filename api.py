"""
RAG API over local documents (txt/md/csv/pdf/images) using Ollama + OCR.

Run:
    pip install -r requirements.txt
    uvicorn api:app --host 0.0.0.0 --port 8000

Docs (auto-generated): http://localhost:8000/docs
"""
import os, glob, pickle, time, threading, hashlib, shutil, json, queue
from contextlib import asynccontextmanager
from typing import Optional, List

import numpy as np
import ollama
from tqdm import tqdm
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------- settings
EMBED = "nomic-embed-text"
CHAT = "llama3.2:3b"
DATA_DIR = "data"
CACHE = "cache.pkl"
TEXT_CACHE_DIR = ".text_cache"
CHUNK_SIZE = 2000
OVERLAP = 150
BATCH = 16
TOP_K = 3
MAX_CHUNKS = None

OCR_DPI = 150
OCR_LANG = "eng"
MIN_TEXT_PER_PAGE = 30

TEXT_TYPES = (".txt", ".md", ".csv")
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
ALL_TYPES = TEXT_TYPES + IMAGE_TYPES + (".pdf",)

SYSTEM_PROMPT = (
    "Answer using only the context provided. "
    "If the answer is not in the context, say you don't know."
)


# ---------------------------------------------------------------- OCR helpers
_ocr_ready = None


def ocr_available():
    global _ocr_ready
    if _ocr_ready is None:
        try:
            import fitz  # noqa: F401
            import pytesseract
            from PIL import Image  # noqa: F401
            pytesseract.get_tesseract_version()
            _ocr_ready = True
        except Exception as e:
            print(f"OCR is not set up ({e}). Install: apt-get install tesseract-ocr && pip install pymupdf pytesseract pillow")
            _ocr_ready = False
    return _ocr_ready


def ocr_image(img):
    import pytesseract
    return pytesseract.image_to_string(img, lang=OCR_LANG)


def read_pdf(path):
    import fitz, io
    from PIL import Image

    doc = fitz.open(path)
    pages, scanned = [], 0
    for page in tqdm(doc, desc=os.path.basename(path)[:45], unit="page", leave=False):
        text = page.get_text() or ""
        if len(text.strip()) < MIN_TEXT_PER_PAGE and ocr_available():
            pix = page.get_pixmap(dpi=OCR_DPI)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = ocr_image(img)
            scanned += 1
        pages.append(text)
    if scanned:
        print(f"  OCR used on {scanned}/{len(doc)} pages: {os.path.basename(path)}")
    return "\n".join(pages)


def read_image(path):
    if not ocr_available():
        return ""
    from PIL import Image
    return ocr_image(Image.open(path))


def file_hash(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_file(path, digest):
    os.makedirs(TEXT_CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(TEXT_CACHE_DIR, digest + ".txt")
    if os.path.exists(cache_path):
        return open(cache_path, encoding="utf-8").read()

    low = path.lower()
    if low.endswith(".pdf"):
        text = read_pdf(path)
    elif low.endswith(IMAGE_TYPES):
        text = read_image(path)
    else:
        text = open(path, encoding="utf-8", errors="ignore").read()

    if text.strip():
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(text)
    return text


# ---------------------------------------------------------------- indexing
def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def list_files():
    return sorted(
        p for p in glob.glob(f"{DATA_DIR}/**/*", recursive=True)
        if os.path.isfile(p) and p.lower().endswith(ALL_TYPES)
    )


def signature(files):
    return [(f, os.path.getsize(f), int(os.path.getmtime(f))) for f in files] + [
        ("settings", CHUNK_SIZE, OVERLAP, MAX_CHUNKS, OCR_DPI, OCR_LANG)
    ]


def build_index(files):
    chunks, seen, empty = [], {}, []
    for n, path in enumerate(files, 1):
        digest = file_hash(path)
        if digest in seen:
            continue
        seen[digest] = path
        print(f"[{n}/{len(files)}] {os.path.basename(path)}", flush=True)
        added = 0
        for c in chunk(read_file(path, digest)):
            if c.strip():
                chunks.append((path, c))
                added += 1
        if added == 0:
            empty.append(path)

    if empty:
        print(f"WARNING: no text could be read from {len(empty)} file(s): {empty[:15]}")
    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    if not chunks:
        raise ValueError("No text found in any file.")

    texts = [c for _, c in chunks]
    out = []
    for i in tqdm(range(0, len(texts), BATCH), desc="Embedding", unit="batch"):
        out += ollama.embed(model=EMBED, input=texts[i:i + BATCH])["embeddings"]
    vecs = np.array(out)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return chunks, vecs


def load_index(force=False):
    files = list_files()
    if not files:
        raise ValueError(f"No supported files found in {DATA_DIR}/ ({', '.join(ALL_TYPES)})")
    sig = signature(files)
    if not force and os.path.exists(CACHE):
        try:
            saved = pickle.load(open(CACHE, "rb"))
            if saved["sig"] == sig:
                return saved["chunks"], saved["vecs"]
        except Exception:
            pass
    chunks, vecs = build_index(files)
    pickle.dump({"sig": sig, "chunks": chunks, "vecs": vecs}, open(CACHE, "wb"))
    return chunks, vecs


# ---------------------------------------------------------------- app state
class State:
    def __init__(self):
        self.chunks: list = []
        self.vecs: Optional[np.ndarray] = None
        self.lock = threading.Lock()          # guards swapping the index
        self.indexing = False
        self.last_error: Optional[str] = None
        self.last_indexed: Optional[float] = None

    @property
    def ready(self):
        return self.vecs is not None and len(self.chunks) > 0


state = State()


def reindex(force=False):
    """Build (or load) the index, then swap it in atomically."""
    if state.indexing:
        return
    state.indexing = True
    state.last_error = None
    try:
        chunks, vecs = load_index(force=force)
        with state.lock:
            state.chunks, state.vecs = chunks, vecs
        state.last_indexed = time.time()
    except Exception as e:
        state.last_error = str(e)
        print(f"Indexing failed: {e}")
    finally:
        state.indexing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DATA_DIR, exist_ok=True)
    # Build in the background so the server starts immediately
    threading.Thread(target=reindex, daemon=True).start()
    yield


app = FastAPI(title="Document Q&A API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten this in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------- schemas
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Chunks passed to the model")
    show_sources: bool = True


class Source(BaseModel):
    file: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source] = []


# ---------------------------------------------------------------- helpers
def require_ready():
    if state.indexing and not state.ready:
        raise HTTPException(503, "Index is still being built. Check /status.")
    if not state.ready:
        raise HTTPException(503, f"Index not available: {state.last_error or 'no documents indexed'}")


def retrieve(question: str, top_k: int):
    qv = np.array(ollama.embed(model=EMBED, input=question)["embeddings"][0])
    qv /= np.linalg.norm(qv)
    with state.lock:
        chunks, vecs = state.chunks, state.vecs
    scores = vecs @ qv
    top = np.argsort(scores)[::-1][:top_k]
    return [(chunks[i][0], chunks[i][1], float(scores[i])) for i in top]


def build_messages(question, hits):
    context = "\n\n".join(f"[{p}]\n{c}" for p, c, _ in hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]


def unique_sources(hits):
    best = {}
    for path, _, score in hits:
        name = os.path.basename(path)
        best[name] = max(score, best.get(name, -1))
    return [Source(file=k, score=round(v, 4)) for k, v in sorted(best.items(), key=lambda x: -x[1])]


# ---------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def status():
    counts = {}
    for path, _ in state.chunks:
        counts[path] = counts.get(path, 0) + 1
    return {
        "ready": state.ready,
        "indexing": state.indexing,
        "chunks": len(state.chunks),
        "files_indexed": len(counts),
        "last_indexed": state.last_indexed,
        "last_error": state.last_error,
        "models": {"embed": EMBED, "chat": CHAT},
    }


@app.get("/files")
def files():
    return {"files": [os.path.relpath(p, DATA_DIR) for p in list_files()]}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """Return the full answer as JSON."""
    require_ready()
    try:
        hits = retrieve(req.question, req.top_k or TOP_K)
        resp = ollama.chat(model=CHAT, messages=build_messages(req.question, hits))
    except Exception as e:
        raise HTTPException(502, f"Model error: {e}")
    return AskResponse(
        answer=resp["message"]["content"],
        sources=unique_sources(hits) if req.show_sources else [],
    )


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """Stream the answer token-by-token as plain text. Sources are sent in the
    X-Sources response header (comma-separated file names)."""
    require_ready()
    try:
        hits = retrieve(req.question, req.top_k or TOP_K)
        messages = build_messages(req.question, hits)
    except Exception as e:
        raise HTTPException(502, f"Model error: {e}")

    def gen():
        try:
            for part in ollama.chat(model=CHAT, messages=messages, stream=True):
                yield part["message"]["content"]
        except Exception as e:
            yield f"\n[error: {e}]"

    headers = {}
    if req.show_sources:
        headers["X-Sources"] = ", ".join(s.file for s in unique_sources(hits))
    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8", headers=headers)


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/ask/events")
def ask_events(req: AskRequest):
    """Stream progress as Server-Sent Events (like the terminal spinner), so the
    connection never sits silent long enough to time out.

    Events:
      status     {"step": "searching"|"thinking", "message": "..."}
      sources    [{"file": "...", "score": 0.73}]        (if show_sources)
      heartbeat  {"step": "...", "elapsed": 12}          (every ~3s while waiting)
      token      {"text": "..."}                         (answer pieces)
      done       {"elapsed": 41}
      error      {"message": "..."}
    """
    require_ready()

    def gen():
        t0 = time.time()
        q = queue.Queue()
        cancelled = threading.Event()
        step = {"name": "searching"}

        def worker():
            try:
                q.put(("status", {"step": "searching", "message": "Searching your documents"}))
                hits = retrieve(req.question, req.top_k or TOP_K)
                if req.show_sources:
                    q.put(("sources", [{"file": s.file, "score": s.score} for s in unique_sources(hits)]))
                step["name"] = "thinking"
                q.put(("status", {"step": "thinking", "message": "Thinking (loading model / generating)"}))
                for part in ollama.chat(model=CHAT, messages=build_messages(req.question, hits), stream=True):
                    if cancelled.is_set():
                        return
                    q.put(("token", {"text": part["message"]["content"]}))
            except Exception as e:
                q.put(("error", {"message": str(e)}))
            finally:
                q.put((None, None))

        threading.Thread(target=worker, daemon=True).start()
        failed = False
        try:
            while True:
                try:
                    ev, data = q.get(timeout=3)
                except queue.Empty:
                    yield sse("heartbeat", {"step": step["name"], "elapsed": int(time.time() - t0)})
                    continue
                if ev is None:
                    break
                if ev == "error":
                    failed = True
                yield sse(ev, data)
            if not failed:
                yield sse("done", {"elapsed": int(time.time() - t0)})
        finally:
            cancelled.set()   # client disconnected or finished: stop generating

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"}
    return StreamingResponse(gen(), media_type="text/event-stream", headers=headers)


@app.post("/reindex", status_code=202)
def reindex_endpoint(background: BackgroundTasks, force: bool = False):
    """Rebuild the index in the background. Use force=true to ignore the cache."""
    if state.indexing:
        raise HTTPException(409, "Indexing already in progress.")
    background.add_task(reindex, force)
    return {"message": "Reindexing started. Poll /status."}


@app.post("/upload", status_code=202)
def upload(background: BackgroundTasks, files: List[UploadFile] = File(...)):
    """Upload one or more documents into the data folder and reindex."""
    if state.indexing:
        raise HTTPException(409, "Indexing in progress; try again when it finishes.")
    saved, rejected = [], []
    os.makedirs(DATA_DIR, exist_ok=True)
    for f in files:
        name = os.path.basename(f.filename or "")
        if not name.lower().endswith(ALL_TYPES):
            rejected.append(name)
            continue
        with open(os.path.join(DATA_DIR, name), "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)
    if saved:
        background.add_task(reindex, False)
    return {"saved": saved, "rejected_unsupported": rejected,
            "message": "Reindexing started." if saved else "Nothing to index."}