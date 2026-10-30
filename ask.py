import os, glob, pickle, sys, time, threading, hashlib
import numpy as np
import ollama

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("Missing library. Run: pip install tqdm")

# ---------------------------------------------------------------- settings
EMBED = "nomic-embed-text"
CHAT = "llama3.2:3b"
DATA_DIR = "data"
CACHE = "cache.pkl"            # saved embeddings index
TEXT_CACHE_DIR = ".text_cache"  # saved text of each file (so OCR is done only once)
CHUNK_SIZE = 2000      # characters per chunk (bigger = fewer chunks = faster to build)
OVERLAP = 150
BATCH = 16             # chunks embedded per call
TOP_K = 3              # chunks passed to the model per question
MAX_CHUNKS = None      # set to e.g. 200 to test quickly on part of the data
SHOW_SOURCES = True

OCR_DPI = 150          # higher (200-300) = better accuracy on poor scans, but slower
OCR_LANG = "eng"       # e.g. "eng+hin" if you installed the Hindi language pack
MIN_TEXT_PER_PAGE = 30  # a page with fewer characters than this is treated as a scan

TEXT_TYPES = (".txt", ".md", ".csv")
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")
ALL_TYPES = TEXT_TYPES + IMAGE_TYPES + (".pdf",)


# ---------------------------------------------------------------- spinner
class Spinner:
    """Animated loader shown while waiting. Call stop() when done."""

    def __init__(self, message):
        self.message = message
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._start_time = time.time()
        self._running = False

    def _run(self):
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        i = 0
        while not self._stop.is_set():
            elapsed = int(time.time() - self._start_time)
            sys.stdout.write(f"\r{frames[i % len(frames)]} {self.message} ({elapsed}s)")
            sys.stdout.flush()
            i += 1
            time.sleep(0.1)

    def start(self):
        self._running = True
        self._thread.start()
        return self

    def stop(self):
        if not self._running:
            return
        self._running = False
        self._stop.set()
        self._thread.join()
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()


# ---------------------------------------------------------------- OCR helpers
_ocr_ready = None


def ocr_available():
    """Check once that the OCR libraries and the tesseract program are installed."""
    global _ocr_ready
    if _ocr_ready is None:
        try:
            import fitz  # noqa: F401  (PyMuPDF)
            import pytesseract
            from PIL import Image  # noqa: F401
            pytesseract.get_tesseract_version()
            _ocr_ready = True
        except Exception as e:
            print(f"\nOCR is not set up ({e}).")
            print("Run:  sudo apt-get install -y tesseract-ocr  &&  pip install pymupdf pytesseract pillow")
            _ocr_ready = False
    return _ocr_ready


def ocr_image(img):
    import pytesseract
    return pytesseract.image_to_string(img, lang=OCR_LANG)


def read_pdf(path):
    """Read a PDF page by page: use the text layer if present, otherwise OCR the page image."""
    import fitz
    from PIL import Image
    import io

    doc = fitz.open(path)
    pages, scanned = [], 0
    for page in tqdm(doc, desc=os.path.basename(path)[:45], unit="page", leave=False):
        text = page.get_text() or ""
        if len(text.strip()) < MIN_TEXT_PER_PAGE:
            if ocr_available():
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
    """Return the text of a file. The result is saved, so OCR runs only once per file."""
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

    # Only save when we got something, so a failed/unconfigured OCR run is retried later
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


def report_skipped():
    supported = set(list_files())
    for p in glob.glob(f"{DATA_DIR}/**/*", recursive=True):
        if os.path.isfile(p) and p not in supported:
            print(f"SKIPPED (unsupported type): {p}")


def signature(files):
    # Changes whenever files are added, removed or modified, so the cache refreshes itself
    return [(f, os.path.getsize(f), int(os.path.getmtime(f))) for f in files] + [
        ("settings", CHUNK_SIZE, OVERLAP, MAX_CHUNKS, OCR_DPI, OCR_LANG)
    ]


def build_index(files):
    chunks, seen, empty = [], {}, []
    print(f"Reading {len(files)} files (scanned pages are OCR'd; this is slow the first time) ...")
    for n, path in enumerate(files, 1):
        digest = file_hash(path)
        if digest in seen:
            print(f"[{n}/{len(files)}] duplicate of {os.path.basename(seen[digest])}, skipping: {os.path.basename(path)}")
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
        print(f"\nWARNING: no text could be read from {len(empty)} file(s):")
        for p in empty[:15]:
            print("   ", p)
        if len(empty) > 15:
            print(f"    ... and {len(empty) - 15} more")

    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    if not chunks:
        raise SystemExit("No text found in any file.")

    print(f"\nEmbedding {len(chunks)} chunks (one-time, result is saved) ...", flush=True)
    texts = [c for _, c in chunks]
    out = []
    with tqdm(total=len(texts), desc="Embedding", unit="chunk") as bar:
        for i in range(0, len(texts), BATCH):
            batch = texts[i:i + BATCH]
            out += ollama.embed(model=EMBED, input=batch)["embeddings"]
            bar.update(len(batch))
    vecs = np.array(out)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return chunks, vecs


def load_index():
    report_skipped()
    files = list_files()
    if not files:
        raise SystemExit(f"No files found in {DATA_DIR}/ ({', '.join(ALL_TYPES)})")
    sig = signature(files)
    if os.path.exists(CACHE):
        try:
            saved = pickle.load(open(CACHE, "rb"))
            if saved["sig"] == sig:
                print("Loaded saved index.", flush=True)
                return saved["chunks"], saved["vecs"]
        except Exception:
            pass
    chunks, vecs = build_index(files)
    pickle.dump({"sig": sig, "chunks": chunks, "vecs": vecs}, open(CACHE, "wb"))
    return chunks, vecs


# ---------------------------------------------------------------- chat loop
def main():
    chunks, vecs = load_index()
    print(f"Ready: {len(chunks)} chunks from your files.")
    counts = {}
    for path, _ in chunks:
        counts[path] = counts.get(path, 0) + 1
    print(f"  ({len(counts)} files contributed text)")
    if MAX_CHUNKS:
        print(f"  NOTE: MAX_CHUNKS is set to {MAX_CHUNKS}, so data is being cut off.")

    while True:
        q = input("\nAsk (or 'quit'): ").strip()
        if not q:
            continue
        if q.lower() in ("quit", "exit"):
            break

        spinner = Spinner("Searching your documents").start()
        qv = np.array(ollama.embed(model=EMBED, input=q)["embeddings"][0])
        qv /= np.linalg.norm(qv)
        top = np.argsort(vecs @ qv)[::-1][:TOP_K]
        spinner.stop()
        context = "\n\n".join(f"[{chunks[i][0]}]\n{chunks[i][1]}" for i in top)

        messages = [
            {
                "role": "system",
                "content": "Answer using only the context provided. "
                           "If the answer is not in the context, say you don't know.",
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {q}"},
        ]
        spinner = Spinner("Thinking").start()
        try:
            for part in ollama.chat(model=CHAT, messages=messages, stream=True):
                spinner.stop()
                print(part["message"]["content"], end="", flush=True)
        finally:
            spinner.stop()
        print()

        if SHOW_SOURCES:
            print("\nSources used: " + ", ".join(sorted({os.path.basename(chunks[i][0]) for i in top})))


if __name__ == "__main__":
    main()