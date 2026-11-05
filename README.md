# Local LLM: ask questions about my own documents (Linux, REST API)

A small "chat with your documents" service that runs fully on a Linux machine
using Ollama. It reads the files in `data/` (including scanned PDFs, using OCR)
and answers questions using only the most relevant parts of those files.
It is exposed as a REST API (FastAPI), so any app, script or front end can use it.
Nothing is sent to the internet after the models are downloaded.

```
files in data/  ->  text (OCR if scanned)  ->  chunks  ->  embeddings  ->  search  ->  llama3.2 answers
                                                                              ^
                                                              POST /ask  (HTTP + JSON)
```

Tested on Debian/Ubuntu-style systems. Commands for Fedora and Arch are included.
These same steps also work inside GitHub Codespaces (see section 10).

---

## 1. Requirements

- Linux (x86_64 or arm64) with `sudo` access
- Python 3.10 or newer
- RAM: 8 GB minimum (3B model), 16 GB recommended
- Disk: about 5 GB free (models are about 2.3 GB, plus your documents and cache)
- A GPU is optional. Without one, answers take longer (see section 9)

## 2. First-time setup

### 2.0 Quick way (Debian/Ubuntu, or macOS with Homebrew)

Install Ollama first (section 2.2), then from the project folder:

```bash
chmod +x setup.sh run.sh
./setup.sh
```

`setup.sh` installs Tesseract, creates `.venv`, installs `requirements.txt`, creates `data/`
and pulls both models. If you use it, skip to section 2.5. Otherwise follow the manual
steps below.

### 2.1 Install system packages

Debian / Ubuntu / Mint:
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv python3-pip tesseract-ocr curl
```

Fedora:
```bash
sudo dnf install -y python3 python3-pip tesseract curl
```

Arch / Manjaro:
```bash
sudo pacman -S --needed python python-pip tesseract tesseract-data-eng curl
```

### 2.2 Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

On systems with systemd, this installs Ollama as a background service that starts
automatically on boot. Check it with:
```bash
systemctl status ollama
```
If you do not use systemd (containers, WSL without systemd, Codespaces), start it by hand:
```bash
ollama serve > /tmp/ollama.log 2>&1 &
```

### 2.3 Download the models

```bash
ollama pull nomic-embed-text     # used to search the documents (~270 MB)
ollama pull llama3.2:3b          # the chat model (~2 GB)
```

### 2.4 Create a Python virtual environment and install libraries

Using a virtual environment avoids the "externally-managed-environment" error
on modern Linux distributions.

```bash
cd ~/local-llm                     # your project folder
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` contains: `fastapi`, `uvicorn[standard]`, `python-multipart`, `ollama`,
`numpy`, `tqdm`, `pymupdf`, `pytesseract`, `pillow`.

You must run `source .venv/bin/activate` in every new terminal before starting the server
(`run.sh` does this for you).

### 2.5 Add your documents

```bash
mkdir -p data
```
Copy your files into `data/` (or upload them later through the API, see section 4).

Supported: `.pdf` (text or scanned), `.txt`, `.md`, `.csv`, and images
(`.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`).
Other types (`.docx`, `.xlsx`) are ignored. Convert them to PDF first.

## 3. Start the server

```bash
./run.sh
```

or manually:

```bash
cd ~/local-llm
source .venv/bin/activate
uvicorn api:app --host 0.0.0.0 --port 8000
```

- The server starts immediately and builds the index **in the background**.
  The first build is slow: it OCRs scanned pages, then embeds every chunk (progress
  bars appear in the terminal). After that, results are saved and startup takes seconds.
- Until the index is ready, `/ask` answers with HTTP 503. Check progress at
  `http://localhost:8000/status` (`"ready": true` means you can ask questions).
- Interactive documentation (try every endpoint in the browser):
  `http://localhost:8000/docs`
- Use `--host 127.0.0.1` instead of `0.0.0.0` if the API should only be reachable from
  the same machine.
- Run a **single worker** (the default). The index is held in memory, so do not use
  `--workers 2` or more.
- Stop the server with `Ctrl+C`.

## 4. API reference

Base URL: `http://localhost:8000`

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check. Returns `{"status": "ok"}` |
| GET | `/status` | Index state: ready, indexing, chunk and file counts, last error, models |
| GET | `/files` | Supported files currently in `data/` |
| POST | `/ask` | Ask a question, get the full answer as JSON |
| POST | `/ask/stream` | Ask a question, get the answer streamed as plain text |
| POST | `/upload` | Upload documents into `data/` and reindex |
| POST | `/reindex` | Rebuild the index in the background (`?force=true` ignores the cache) |

### POST /ask

Request body:

| Field | Type | Default | Meaning |
|---|---|---|---|
| `question` | string | required | Your question |
| `top_k` | integer 1-20 | `TOP_K` setting (3) | Chunks sent to the model for this question |
| `show_sources` | boolean | `true` | Include the source files in the response |

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Under Section 16, what are the conditions for ITC?", "top_k": 4}'
```

Response:
```json
{
  "answer": "...",
  "sources": [
    {"file": "gst_act.pdf", "score": 0.7312},
    {"file": "circular_12.pdf", "score": 0.6841}
  ]
}
```

`score` is the cosine similarity of the best-matching chunk from that file (higher is closer).

### POST /ask/stream

Same request body as `/ask`. The answer arrives token by token as `text/plain`.
The source file names are sent in the `X-Sources` response header.

```bash
curl -N -i -X POST http://localhost:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the penalty provisions"}'
```

### POST /upload

Multipart upload. Files with unsupported types are rejected and listed in the response.
Uploading a file with the same name as an existing one overwrites it. Returns 202 and
reindexes in the background (poll `/status`).

```bash
curl -X POST http://localhost:8000/upload \
  -F "files=@/path/to/file1.pdf" \
  -F "files=@/path/to/scan.png"
```

### POST /reindex

```bash
curl -X POST http://localhost:8000/reindex                # use cache where files are unchanged
curl -X POST "http://localhost:8000/reindex?force=true"   # re-embed everything
```

Returns 202 immediately. Returns 409 if indexing is already running. While a reindex runs,
`/ask` keeps answering from the previous index, and the new one is swapped in when ready.

### Calling it from Python

```python
import requests

r = requests.post("http://localhost:8000/ask",
                  json={"question": "What is the due date for GSTR-3B?"})
data = r.json()
print(data["answer"])
print(data["sources"])
```

### Status codes

| Code | Meaning |
|---|---|
| 200 | OK |
| 202 | Accepted: upload or reindex started in the background |
| 409 | Indexing already in progress |
| 422 | Invalid request body (for example, empty question) |
| 502 | Ollama or model error while answering |
| 503 | Index not ready yet, or no documents could be indexed (see `/status`) |

## 5. Every time you come back

1. Make sure Ollama is running:
   ```bash
   systemctl is-active ollama       # should print "active"
   ```
   If it is not active: `sudo systemctl start ollama`
   (no systemd: `ollama serve > /tmp/ollama.log 2>&1 &`, then wait 5 seconds)
2. Check the models are still there: `ollama list`
3. Start the server: `./run.sh`
4. Wait for `"ready": true` at `http://localhost:8000/status`.

## 6. Adding or changing data

Option A: through the API
```bash
curl -X POST http://localhost:8000/upload -F "files=@newfile.pdf"
```

Option B: on disk
1. Put the new files into `data/`.
2. Call `POST /reindex` (or restart the server). Changes are detected automatically.

Forcing a rebuild:
- Full re-embed: `POST /reindex?force=true`, or stop the server and `rm cache.pkl`
- Full re-OCR of everything: stop the server, then `rm -r .text_cache cache.pkl` and restart

Identical duplicate files are detected and read only once.

## 7. Project files

| File / folder | What it is |
|---|---|
| `api.py` | The whole service: OCR, chunking, embeddings, search, chat, REST endpoints |
| `requirements.txt` | Python packages |
| `setup.sh` | One-time install script (Debian/Ubuntu, macOS with Homebrew) |
| `run.sh` | Activates `.venv` and starts the server on port 8000 |
| `data/` | Your documents |
| `.venv/` | Python virtual environment |
| `.text_cache/` | Text extracted from each file (OCR is slow, so it is saved here) |
| `cache.pkl` | Saved embeddings index (makes startup fast) |
| `chunk_report.txt` | Optional report created by the command in section 8 |

```
local-llm/
├── api.py
├── requirements.txt
├── setup.sh
├── run.sh
├── README.md
├── data/
├── .venv/
├── .text_cache/        (created automatically)
├── cache.pkl           (created automatically)
└── chunk_report.txt    (optional)
```

Note: the older terminal version (`ask.py`) is no longer needed. If you still keep it, it
shares `cache.pkl` and `.text_cache/` with the API, so do not run both at the same time.

## 8. See what was read (characters and chunks per file)

Run from the project folder with the virtual environment active. It lists every file
with its characters and chunks, shows totals at the top and bottom, and saves the
same output to `chunk_report.txt`. It reads the saved text cache, so the server must have
indexed the files at least once.

```bash
python - <<'EOF'
import os, api

rows, seen = [], {}
for f in api.list_files():
    d = api.file_hash(f)
    txt = os.path.join(api.TEXT_CACHE_DIR, d + ".txt")
    dup = d in seen
    seen.setdefault(d, f)
    if os.path.exists(txt):
        text = open(txt, encoding="utf-8").read()
        chars = len(text)
        chunks = sum(1 for c in api.chunk(text) if c.strip())
        note = txt
    else:
        chars = chunks = 0
        note = "NO TEXT (not read/OCR'd yet)"
    if dup:
        note = "DUPLICATE of " + os.path.basename(seen[d]) + " (not counted in totals)"
    rows.append((chunks, chars, f, note, dup))

uniq = [r for r in rows if not r[4]]
t_chunks = sum(r[0] for r in uniq)
t_chars = sum(r[1] for r in uniq)

lines = [
    f"TOTAL: {t_chars:,} characters | {t_chunks:,} chunks | {len(uniq)} unique files ({len(rows)} files in data/)",
    "-" * 100,
    f"{'chunks':>7} {'characters':>12}  file  [text file / note]",
]
for chunks, chars, f, note, dup in sorted(rows, key=lambda r: -r[1]):
    lines.append(f"{chunks:7d} {chars:12,d}  {os.path.basename(f)}  [{note}]")
lines += ["-" * 100, f"TOTAL: {t_chars:,} characters | {t_chunks:,} chunks"]

out = "\n".join(lines)
print(out)
open("chunk_report.txt", "w", encoding="utf-8").write(out)
EOF
```

To see which files in `data/` are ignored because of their type, compare `ls -R data/`
with `GET /files`.

## 9. Settings (top of `api.py`)

Settings are Python constants. Restart the server after changing them.

| Setting | Meaning | Notes |
|---|---|---|
| `CHAT` | Chat model | `llama3.2:3b` is the default. `llama3.2:1b` is about 2x faster but less accurate. With 16 GB+ RAM or a GPU, try `qwen2.5:7b` or `llama3.1:8b` for better answers |
| `TOP_K` | Default chunks sent to the model per question | 3 is fast. 4-5 gives better recall but is slower. Callers can override it per request with `top_k`. Changing it does not rebuild the index |
| `CHUNK_SIZE` | Characters per chunk | Bigger = fewer chunks = faster to build. Changing it rebuilds the index |
| `OCR_DPI` | Scan quality for OCR | 150 default. Use 200-300 for poor scans (slower) |
| `OCR_LANG` | OCR language | `eng`. For Hindi install the language pack (below) and set `"eng+hin"` |
| `MAX_CHUNKS` | Limit chunks for quick tests | Keep `None` for normal use |
| `SYSTEM_PROMPT` | Instruction given to the model | Default: answer only from the context, otherwise say you don't know |

Source display is controlled per request with `show_sources` (the old `SHOW_SOURCES`
constant no longer exists).

Extra OCR languages:
- Debian/Ubuntu: `sudo apt-get install -y tesseract-ocr-hin`
- Fedora: `sudo dnf install -y tesseract-langpack-hin`
- Arch: `sudo pacman -S tesseract-data-hin`

Speed notes:
- With a supported NVIDIA or AMD GPU, Ollama uses it automatically and answers are much faster.
- Check what Ollama is using with `ollama ps` (shows CPU or GPU).
- Watch Ollama's errors with `journalctl -u ollama -n 50 --no-pager`.
- `/ask/stream` feels faster on slow machines because words appear as they are generated.

## 10. Troubleshooting

| Problem | Fix |
|---|---|
| `/ask` returns 503 "Index is still being built" | Wait. Check `GET /status`. The first build with OCR can take a long time |
| `/ask` returns 503 "Index not available" | See `last_error` in `/status`. Usually `data/` is empty or has no readable text |
| `/ask` returns 502 "Model error" / `Failed to connect to Ollama` | Start it: `sudo systemctl start ollama` (no systemd: `ollama serve > /tmp/ollama.log 2>&1 &`, wait 5 seconds). Then check `ollama list` shows both models |
| `externally-managed-environment` when running pip | You are not in the virtual environment. Run `source .venv/bin/activate` first |
| `ModuleNotFoundError` | Activate the virtual environment, then `pip install -r requirements.txt` |
| `Form data requires "python-multipart"` | `pip install python-multipart` (it is in `requirements.txt`) |
| `Address already in use` | Port 8000 is taken. Use another: `uvicorn api:app --port 8001`, or stop the other process |
| `/upload` or `/reindex` returns 409 | Indexing is already running. Wait until `/status` shows `"indexing": false` |
| "OCR is not set up" in the server log | Install Tesseract (section 2.1) and the pip libraries (section 2.4), then `POST /reindex?force=true`. Files read without OCR are not cached, so they are retried |
| Stuck waiting for an answer for minutes | CPU-only machines are slow. Set `TOP_K = 2`, use `llama3.2:1b`, or check `ollama ps` and `journalctl -u ollama -n 50` |
| A file adds no chunks | It has no readable text. Check the WARNING list in the server log, or run the report in section 8 |
| Garbled OCR text | Raise `OCR_DPI` to 200 or 300, stop the server, `rm -r .text_cache cache.pkl`, start again |
| Wrong or incomplete answers | Raise `top_k` to 5, mention the section or topic in the question, and always verify important answers in the source PDF |
| Indexing was interrupted | OCR text is kept in `.text_cache`, but embeddings are only saved at the end, so restart the server or call `/reindex` |
| Out of memory or very slow | Close other programs, or switch to `llama3.2:1b` |
| Browser app gets CORS errors | CORS is open to all origins by default. If you changed it, allow your app's origin in `allow_origins` in `api.py` |

Using GitHub Codespaces instead of your own machine: the same steps work, but start
Ollama by hand after each restart (`ollama serve > /tmp/ollama.log 2>&1 &`), keep
the browser tab open during long runs, and stop the codespace when done. Forward port 8000
from the Ports tab to reach the API from your browser. The free plan gives about
120 core-hours per month.

## 11. Security and deployment

- The API has **no authentication**. Anyone who can reach the port can ask questions about your
  documents and upload files. Keep it on `127.0.0.1` or a private network, or put it behind a
  reverse proxy (nginx, Caddy) that adds authentication and HTTPS.
- CORS is set to allow all origins (`allow_origins=["*"]` in `api.py`). Restrict it to your
  front end's address before exposing the service.
- `/upload` accepts only the supported file types and uses only the file's base name, but it
  has no size limit. Add one at the proxy if untrusted users can reach it.
- Keep confidential documents on a machine you are allowed to store them on, and check your
  company's policy.

Run it as a service on boot (systemd). Create `/etc/systemd/system/doc-api.service`
(adjust the user and paths):
```ini
[Unit]
Description=Document Q&A API
After=network.target ollama.service

[Service]
User=youruser
WorkingDirectory=/home/youruser/local-llm
ExecStart=/home/youruser/local-llm/.venv/bin/uvicorn api:app --host 127.0.0.1 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
Then:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now doc-api
journalctl -u doc-api -f
```

## 12. Limits to remember

- Each answer can take from seconds (GPU) to a couple of minutes (slow CPU).
- Requests are handled one after another by the model; several simultaneous questions will queue
  and each will be slower.
- Each question is independent: the API does not remember earlier questions (no chat history).
- The model only sees the top few chunks per question, so it is weak at "summarize everything" or "list all..." questions.
- A 3B model can mix up numbers, dates and conditions in legal text. Treat answers as a quick guide and check the source.
- The documents are a snapshot. Newer amendments are not known unless you add them to `data/`.

## 13. Optional: do not commit generated files

Add this to `.gitignore` if the project is in a git repository:
```
.venv/
.text_cache/
cache.pkl
chunk_report.txt
data/
__pycache__/
```