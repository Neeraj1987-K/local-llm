# Local LLM: ask questions about my own documents (GST files)

A small local "chat with your documents" project that runs inside GitHub Codespaces
using Ollama. It reads the files in `data/` (including scanned PDFs, using OCR),
and answers questions using only the most relevant parts of those files.

```
files in data/  ->  text (OCR if scanned)  ->  chunks  ->  embeddings  ->  search  ->  llama3.2 answers
```

---

## 1. First-time setup (new codespace)

Run these in the Codespaces terminal (open with `` Ctrl+` ``).

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Install the OCR program (reads scanned pages)
sudo apt-get update && sudo apt-get install -y tesseract-ocr

# 3. Install the Python libraries
pip install ollama numpy tqdm pymupdf pytesseract pillow

# 4. Start the Ollama server in the background
ollama serve > /tmp/ollama.log 2>&1 &

# 5. Download the models (wait ~5 seconds after step 4)
ollama pull nomic-embed-text     # used to search the documents (~270 MB)
ollama pull llama3.2:3b          # the chat model (~2 GB)
```

Then put your files in the `data/` folder (create it with `mkdir -p data`).

Supported files: `.pdf` (text or scanned), `.txt`, `.md`, `.csv`,
and images (`.png`, `.jpg`, `.jpeg`, `.tif`, `.bmp`).
Other types (`.docx`, `.xlsx`) are skipped. Convert them to PDF first.

## 2. Run it

```bash
python ask.py
```

- First run is slow: it OCRs scanned pages, then embeds every chunk (progress bars are shown).
- After that, the results are saved and it starts in a few seconds.
- Type a question at the `Ask` prompt. Type `quit` to exit.
- Good questions are specific: "Under Section 16, what are the conditions for ITC?"
  Avoid "hi" or vague questions, because the script still searches the documents for them.

## 3. Every time the codespace restarts

Background programs stop when the codespace goes idle or restarts.
If you see `ConnectionError: Failed to connect to Ollama`, run:

```bash
ollama serve > /tmp/ollama.log 2>&1 &
sleep 5
ollama list          # check nomic-embed-text and llama3.2:3b are still there
python ask.py
```

If the models are missing, pull them again (step 5 above).
If `ollama: command not found`, redo the whole first-time setup.

## 4. Adding or changing data

1. Drop the new files into `data/` (the folder, not the project root).
2. Run `python ask.py`. It notices changes by itself and rebuilds the index.
3. To force a full re-embed: `rm cache.pkl`
4. To force a re-OCR of everything: `rm -r .text_cache cache.pkl`

Identical duplicate files are detected and read only once.

## 5. Project files

| File / folder | What it is |
|---|---|
| `ask.py` | The whole program: OCR, chunking, embeddings, search, chat |
| `data/` | Your documents |
| `.text_cache/` | Text extracted from each file (OCR is slow, so it is saved here) |
| `cache.pkl` | Saved embeddings index (makes startup fast) |
| `chunk_report.txt` | Optional report created by the command in section 7 |

## 6. Settings (top of `ask.py`)

| Setting | Meaning | Notes |
|---|---|---|
| `CHAT` | Chat model | `llama3.2:3b` is the default. `llama3.2:1b` is about 2x faster but less accurate |
| `TOP_K` | Chunks sent to the model per question | 3 is fast. 4-5 gives better recall but is slower. Changing it does not rebuild the index |
| `CHUNK_SIZE` | Characters per chunk | Bigger = fewer chunks = faster to build. Changing it rebuilds the index |
| `OCR_DPI` | Scan quality for OCR | 150 default. Use 200-300 for poor scans (slower) |
| `OCR_LANG` | OCR language | `eng`. For Hindi: `sudo apt-get install -y tesseract-ocr-hin` then set `"eng+hin"` |
| `MAX_CHUNKS` | Limit chunks for quick tests | Keep `None` for normal use |

## 7. See what was read (characters and chunks per file)

Run from the project folder. Lists every file with its characters and chunks,
the totals at the top and bottom, and saves the same to `chunk_report.txt`.

```bash
python - <<'EOF'
import os, ask

rows, seen = [], {}
for f in ask.list_files():
    d = ask.file_hash(f)
    txt = os.path.join(ask.TEXT_CACHE_DIR, d + ".txt")
    dup = d in seen
    seen.setdefault(d, f)
    if os.path.exists(txt):
        text = open(txt, encoding="utf-8").read()
        chars = len(text)
        chunks = sum(1 for c in ask.chunk(text) if c.strip())
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

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Failed to connect to Ollama` | Start it: `ollama serve > /tmp/ollama.log 2>&1 &` then wait 5 seconds |
| Stuck on "Thinking" for minutes | The 2-core machine is slow. Set `TOP_K = 2`, or use `llama3.2:1b`. Check `ollama ps` and `tail -20 /tmp/ollama.log` |
| "OCR is not set up" | `sudo apt-get install -y tesseract-ocr` and `pip install pymupdf pytesseract pillow` |
| A file adds no chunks | It has no readable text. Check the WARNING list printed after reading, or run the report in section 7 |
| Garbled OCR text | Raise `OCR_DPI` to 200 or 300, delete `.text_cache`, run again |
| Wrong or incomplete answers | Raise `TOP_K` to 5, mention the section or topic in the question, and always verify important answers in the source PDF |
| Embedding was interrupted | The OCR text is kept in `.text_cache`, but embeddings are only saved at the end, so rerun `python ask.py` |

## 9. Codespaces tips (free plan)

- Free accounts get about 120 core-hours per month. A 2-core machine uses 2 core-hours per hour.
- Stop the codespace when you are done (github.com/codespaces), because idle time still counts until it times out.
- Models, `.text_cache` and `cache.pkl` live inside the codespace. If you delete the codespace, they are lost and must be rebuilt (OCR and embedding take the longest).
- Do not put confidential work documents into a personal codespace without checking your company's policy.

## 10. Limits to remember

- Runs on CPU only, so each answer can take from about 30 seconds to a couple of minutes.
- The model only sees the top few chunks per question, so it is weak at "summarize everything" or "list all..." questions.
- A 3B model can mix up numbers, dates and conditions in legal text. Treat answers as a quick guide and check the source.
- The documents are a snapshot. Newer amendments are not known unless you add them to `data/`.