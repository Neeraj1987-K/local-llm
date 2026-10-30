import os, glob, pickle, sys, time, threading
import numpy as np
import ollama

try:
    from tqdm import tqdm
except ImportError:
    raise SystemExit("Missing progress bar library. Run: pip install tqdm")

EMBED = "nomic-embed-text"
CHAT = "llama3.2:3b"
DATA_DIR = "data"
CACHE = "cache.pkl"
CHUNK_SIZE = 1000      # characters per chunk
OVERLAP = 100
BATCH = 16             # chunks embedded per call (shows progress)
TOP_K = 3              # chunks passed to the model per question
MAX_CHUNKS = None      # set to e.g. 200 to test quickly on part of the data
SHOW_SOURCES = True


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
        sys.stdout.write("\r" + " " * 70 + "\r")  # clear the spinner line
        sys.stdout.flush()


def read_file(path):
    if path.endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    return open(path, encoding="utf-8", errors="ignore").read()


def chunk(text, size=CHUNK_SIZE, overlap=OVERLAP):
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i + size])
        i += size - overlap
    return out


def list_files():
    return sorted(
        p for p in glob.glob(f"{DATA_DIR}/**/*", recursive=True)
        if os.path.isfile(p) and p.endswith((".txt", ".md", ".csv", ".pdf"))
    )


def signature(files):
    # Changes whenever files are added, removed or modified, so the cache refreshes itself
    return [(f, os.path.getsize(f), int(os.path.getmtime(f))) for f in files] + [
        ("settings", CHUNK_SIZE, OVERLAP, MAX_CHUNKS)
    ]


def build_index(files):
    chunks = []
    for path in files:
        print(f"Reading {path} ...", flush=True)
        for c in chunk(read_file(path)):
            if c.strip():
                chunks.append((path, c))
    if MAX_CHUNKS:
        chunks = chunks[:MAX_CHUNKS]
    if not chunks:
        raise SystemExit("No text found. If your PDF is scanned, run OCR on it first.")

    print(f"Embedding {len(chunks)} chunks (one-time, result is saved) ...", flush=True)
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
    files = list_files()
    if not files:
        raise SystemExit(f"No files found in {DATA_DIR}/ (.txt, .md, .csv, .pdf)")
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


def main():
    chunks, vecs = load_index()
    print(f"Ready: {len(chunks)} chunks from your files.")

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
                spinner.stop()  # first word arrived, hide the loader
                print(part["message"]["content"], end="", flush=True)
        finally:
            spinner.stop()
        print()

        if SHOW_SOURCES:
            print("\nSources used: " + ", ".join(sorted({chunks[i][0] for i in top})))


if __name__ == "__main__":
    main()