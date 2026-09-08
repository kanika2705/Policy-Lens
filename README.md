# PolicyLens — AI Policy Impact Analyzer

PolicyLens lets you upload a policy document (a PDF — think attendance rules,
scholarship eligibility, admission criteria) and ask questions like *"I have
attended 32 out of 45 classes, what happens if I miss 3 more?"* It retrieves
the relevant part of the policy, answers using **only** that text, shows you
the exact source excerpt it used, and does any required math in Python
(not the LLM) so the numbers are trustworthy.

A second tab lets you upload two versions of a policy (old vs. new) and see
what changed, categorized as Added / Removed / Modified, with a note on who
each change affects.

This is a student portfolio project — the goal is a clean, correct,
explainable MVP, not a feature-packed product.

---

## 1. Setup — 100% free, runs entirely on your own machine

PolicyLens needs **no API key, no billing, and no paid service of any kind.**
Everything — embeddings and the LLM — runs locally:

- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`), a small model
  that runs on your CPU.
- **LLM**: [Ollama](https://ollama.com), running a small local model
  (`llama3.2:3b` by default) on your own machine.

### What you need to install

1. **Python 3.10+** — [python.org/downloads](https://www.python.org/downloads/)
   (if you don't already have it).
2. **Ollama** — the free local LLM runner. Instructions below.
3. **This project's Python packages** — via `requirements.txt`, below.

### Step 1 — Install Ollama (Windows)

1. Go to **https://ollama.com/download** and click **Download for Windows**.
2. Run the installer (`OllamaSetup.exe`) and follow the prompts — it's a
   normal Windows installer, next-next-finish.
3. Once installed, Ollama runs automatically as a background service and
   starts a local server at `http://localhost:11434`. You'll see a small
   llama icon in your system tray (bottom-right, near the clock) when it's
   running.

You do **not** need to keep a terminal window open for Ollama itself — it
runs in the background once installed, and restarts automatically when your
laptop reboots.

### Step 2 — Download the model

Open **Command Prompt** (or PowerShell) and run:

```bash
ollama pull llama3.2:3b
```

This downloads the model (a few GB) **once**. After that, it's stored
locally and works completely offline. `llama3.2:3b` is a good default: small
enough to run on a normal laptop with no GPU, capable enough to follow the
"answer only from this text" instructions this project relies on.

> If your laptop is lower-spec (e.g. 8GB RAM) and `llama3.2:3b` feels slow,
> try the smaller `llama3.2:1b` instead:
> `ollama pull llama3.2:1b`, then set `OLLAMA_MODEL=llama3.2:1b` in your
> `.env` file (see below).

Confirm it downloaded correctly:

```bash
ollama list
```

You should see `llama3.2:3b` in the output.

### Step 3 — Install the Python project

```bash
git clone <this-repo>
cd policylens
python -m venv venv
venv\Scripts\activate          # Windows. Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
```

`sentence-transformers` will pull in PyTorch as a dependency — this is a
sizeable download (roughly 1–2GB) but it's a **one-time, free** install, not
a subscription or API cost.

### Step 4 — (Optional) configure which model to use

```bash
copy .env.example .env
```

The `.env` file only needs to exist if you want to override the default
model (`llama3.2:3b`). No API key goes in it — there isn't one.

### Step 5 — Run the app

```bash
streamlit run app.py
```

Streamlit opens the app in your browser automatically (usually
`http://localhost:8501`).

Two sample policy PDFs are included in `sample_policies/` (an original and a
revised version of a made-up university attendance/scholarship policy) so you
can try the app immediately without hunting for a real PDF.

### Does Ollama need to be running while I use the app?

**Yes.** Ollama needs to be running in the background whenever you use
PolicyLens, because the app sends each question to Ollama's local server to
get an answer. In practice this is automatic: once installed, Ollama starts
itself in the background and stays running (you'll see its icon in the
system tray). You don't need to manually start it each time — but if you
ever see a "Could not connect to Ollama" error in the app, check that the
tray icon is present, or start it from the Start Menu by searching "Ollama".

---

## 2. How to use it

**Tab 1 — Policy Q&A**
1. Upload a policy PDF.
2. Type a question about your specific situation.
3. Click **Analyze Policy**.
4. Read the answer, any Python-calculated numbers, and the source excerpts
   the answer was based on.

**Tab 2 — Policy Comparison**
1. Upload an older and a newer version of the same policy.
2. Click **Compare Policies**.
3. Review the list of changes, each labeled Added / Removed / Modified with
   an explanation of who it affects.

---

## 3. Project structure

```
policylens/
├── app.py                    # Streamlit UI — layout, uploads, session state
├── requirements.txt
├── .env.example
├── README.md
├── src/
│   ├── pdf_processor.py      # PDF -> text -> overlapping chunks (page/section tagged)
│   ├── embeddings.py         # text -> vector, via local Sentence-Transformers
│   ├── retriever.py          # FAISS vector store: build index, similarity search
│   ├── llm.py                 # all LLM calls: grounded answer, rule extraction, change classification
│   ├── policy_analyzer.py    # orchestrates Q&A pipeline + Python calculator
│   └── comparison.py          # orchestrates the policy-diff pipeline
└── sample_policies/           # two sample PDFs (old + new) for demoing
```

Each module has one job. `app.py` never calls FAISS or Ollama directly — it
only calls functions in `src/`. This separation is what makes the pipeline
easy to explain (and easy to test piece by piece).

---

## 4. Architecture — how a question gets answered

```
   PDF upload
       │
       ▼
 pdf_processor.py  ──►  list of chunks (text + page number + section guess)
       │
       ▼
 embeddings.py     ──►  each chunk becomes a vector, locally, via
       │                Sentence-Transformers (all-MiniLM-L6-v2)
       ▼
 retriever.py      ──►  vectors stored in a FAISS index (in memory)


   User question
       │
       ▼
 embeddings.py     ──►  question becomes a vector too (same local model)
       │
       ▼
 retriever.py      ──►  FAISS returns the top-k most similar chunks
       │
       ▼
 llm.py            ──►  local LLM (via Ollama) answers using ONLY those chunks
       │                (also extracts any numeric rules, e.g. "75%")
       ▼
 policy_analyzer.py──►  if the question needs math, Python calculates it
       │                (attendance %, CGPA comparison, etc.)
       ▼
   app.py displays: answer + calculated numbers + source excerpts
```

This whole pattern — retrieve relevant text, then generate an answer from
it — is called **RAG (Retrieval-Augmented Generation)**. Nothing in this
pipeline calls out to a paid API: embeddings run on your CPU via
Sentence-Transformers, and the LLM runs on your CPU (or GPU, if you have one)
via Ollama.

---

## 5. Key concepts explained (for interviews)

**Embeddings**
A way of converting text into a list of numbers (a vector) that represents
its *meaning*. Texts with similar meaning end up as vectors that are close
together, even if they don't share the same words. This project uses a
local, free Sentence-Transformers model, `all-MiniLM-L6-v2`, which produces
384-dimensional vectors and runs entirely on CPU — no API call involved.

**Vector search (FAISS)**
Once every chunk of the policy is a vector, FAISS lets us quickly find
"which stored vectors are closest to this new vector" (the user's question,
also embedded). This project uses cosine similarity (via normalized vectors
and an inner-product index) — a standard choice for text embeddings.

**Retrieval**
The step of using vector search to pull out the small number of chunks
(here, the top 4) that are most relevant to the question, out of potentially
hundreds of chunks in the full document.

**RAG (Retrieval-Augmented Generation)**
The overall pattern: *retrieve* relevant context first, then *generate* an
answer using an LLM that is given that context directly in its prompt. The
LLM never has to "remember" the whole policy — it only has to reason over
the few paragraphs it's handed.

**The LLM's role here**
Two narrow jobs: (1) explain, in plain language, what the retrieved policy
text says in relation to the question, and (2) pull out any explicit numeric
thresholds mentioned (like "75% attendance required") as structured data.
It is explicitly instructed not to use outside knowledge and not to do
arithmetic itself. The LLM itself is `llama3.2:3b`, running locally through
Ollama — a small (~2GB) open-weight model, not a hosted API.

**Ollama**
A free, local tool that downloads and runs open-weight LLMs on your own
computer, and exposes them through a simple local server
(`http://localhost:11434`). Instead of sending your questions to a company's
servers over the internet, your laptop is doing the inference itself. This
project talks to Ollama through its official Python package (`ollama.chat(...)`),
which is really just a thin wrapper around HTTP requests to that local server.

---

## 6. Why RAG instead of just pasting the whole PDF into the prompt?

- **Speed**: a small local model like `llama3.2:3b` is far slower than a
  hosted API at processing long inputs. Sending an entire 20+ page PDF on
  every question would make each answer take much longer (and on a modest
  laptop, could time out or exhaust memory). Retrieval sends only the ~4
  most relevant chunks, keeping each answer fast even on modest hardware.
- **Accuracy over long documents**: LLMs are more reliable when the input is
  short and focused. Studies and practical experience both show that models
  are more likely to miss or misstate details buried in a very long context
  ("lost in the middle") than details in a short, targeted excerpt.
- **Traceable evidence**: because we know exactly which chunks were sent to
  the model, we can show the user precisely which page/section the answer
  came from. If we pasted the whole document in, we'd have to guess at
  after-the-fact what the model actually used.
- **Scales to bigger documents**: this approach works the same whether the
  PDF is 5 pages or 500 — only the indexing step gets bigger, not the cost of
  answering each question.

---

## 7. How this reduces hallucination

Hallucination (an LLM confidently stating something false) is reduced here
through three layers:

1. **Grounding**: the system prompt tells the model to answer *only* from the
   provided excerpts, and to explicitly say "the policy does not specify
   this" when the excerpts don't cover the question, instead of guessing.
2. **Narrow retrieval**: by only including the top few relevant chunks (not
   the whole document, and not the model's general knowledge), there's much
   less room for the model to wander into unrelated or made-up content.
3. **No LLM arithmetic**: any calculation (attendance %, CGPA comparisons) is
   done by plain Python code using numbers the LLM only had to *extract*, not
   compute. This removes the most common source of confidently-wrong numbers
   in LLM outputs.

This does **not** make hallucination impossible — see Limitations below.

---

## 8. Limitations

- **Small local models are less capable than large hosted ones.**
  `llama3.2:3b` is much smaller than models like GPT-4 or Claude, so it can
  occasionally misfollow an instruction (e.g. forget to say "not specified"
  when it should) or phrase things more roughly. This is the trade-off for
  running fully free and offline. If answer quality feels weak, trying a
  different pulled model (`OLLAMA_MODEL` in `.env`) or a larger one your
  machine can handle (e.g. `llama3.1:8b`) usually helps.
- **First response after startup can be slow.** The very first question
  after starting the app (or after Ollama has been idle) can take longer,
  since Ollama loads the model into memory on first use. Subsequent
  questions are faster.
- **Retrieval can miss relevant text.** If the answer depends on information
  spread across many different sections of a long document, the top-k chunks
  might not capture all of it.
- **The Python calculator is scenario-specific.** It currently only handles
  two patterns (attendance %, CGPA comparison) via regular expressions, not
  arbitrary arithmetic. A question involving different math (e.g. GPA-hour
  weighted averages) would fall back to the LLM's plain-text explanation
  without a computed number.
- **PDF text extraction can be imperfect.** Scanned/image-based PDFs, unusual
  layouts, or multi-column formatting can produce messy extracted text, which
  can degrade chunk quality.
- **Section-heading detection is a heuristic**, not a true document-structure
  parser — it may mislabel or miss headings in unusually formatted PDFs.
- **The comparison feature diffs at the paragraph level.** Very reformatted
  documents (heavy reflow, renumbered sections) can produce noisier diffs
  than a genuinely rewritten clause would.
- **Still an LLM.** Even with grounding, the model can occasionally
  misinterpret or summarize retrieved text incorrectly — this system reduces
  hallucination risk, it doesn't eliminate it. Always treat outputs as a
  starting point, not a final legal/administrative decision.

---

## 9. Example questions to try

Using the included `sample_policies/sample_attendance_policy.pdf`:

1. "I have attended 32 out of 45 classes. What happens if I miss 3 more?"
2. "Am I eligible for the Merit Scholarship if my CGPA is 8.2?"
3. "What documents do I need to apply for the scholarship?"
4. "What happens if I miss the scholarship deadline?"
5. "If my attendance is 68%, can I still sit for the exam?"

---

## 10. Possible extensions (research-oriented directions)

1. **Evaluation harness**: build a small labeled test set (question, correct
   answer, correct source page) and measure retrieval precision/recall and
   answer faithfulness automatically — turns this from a demo into something
   you can report metrics on.
2. **Better retrieval**: experiment with hybrid search (combining keyword
   search like BM25 with vector search), re-ranking retrieved chunks with a
   cross-encoder, or chunking strategies that respect document structure
   (e.g. never splitting a numbered clause across chunks).
3. **General-purpose numeric reasoning**: replace the two hand-written
   calculators with an LLM-driven "tool use" approach, where the model
   identifies which values it needs, calls a generic calculator tool with
   those values, and the tool (not the model) performs the arithmetic —
   generalizing beyond attendance/CGPA to arbitrary policy math.
