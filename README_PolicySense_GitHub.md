# PolicySense — AI Policy Impact Analyzer

PolicySense is a fully local GenAI application that helps users understand policy documents and determine how specific rules apply to their situation.

Instead of simply summarizing a PDF, it retrieves relevant policy provisions, generates a grounded response, provides supporting evidence, and performs applicable numerical calculations using Python.

## Features

### Policy Q&A
- Upload a policy PDF.
- Ask questions about a specific situation.
- Retrieve the most relevant policy sections using semantic search.
- Generate an answer using a local LLM.
- Display supporting policy excerpts and page information.
- Perform supported numerical calculations with Python.

### Policy Comparison
- Upload an older and newer version of a policy.
- Identify meaningful changes.
- Categorize changes as **Added**, **Removed**, or **Modified**.
- Explain the potential impact of important changes.

## Example

Given a policy stating that students need at least 75% attendance:

> "I attended 32 out of 45 classes. What happens if I miss 3 more?"

PolicySense retrieves the relevant attendance rule, calculates the attendance percentage programmatically, compares it with the policy requirement, and presents the result with supporting evidence.

## Architecture

```text
PDF
 │
 ▼
PDF Text Extraction
 │
 ▼
Text Chunking
 │
 ▼
Sentence-Transformers
(all-MiniLM-L6-v2)
 │
 ▼
FAISS Vector Index
 │
 │       User Question
 │              │
 │              ▼
 │       Question Embedding
 │              │
 └──────► Similarity Search
                │
                ▼
       Relevant Policy Chunks
                │
                ▼
       Local LLM via Ollama
          (llama3.2:3b)
                │
       ┌────────┴────────┐
       ▼                 ▼
  Grounded Answer    Rule Extraction
       │                 │
       └────────┬────────┘
                ▼
        Python Calculation
                │
                ▼
      Answer + Evidence + Results
```

The application uses **RAG (Retrieval-Augmented Generation)**: relevant information is retrieved from the policy first, and the LLM generates its response using that retrieved context.

## Tech Stack

| Component | Technology |
|---|---|
| Interface | Streamlit |
| Language | Python |
| PDF processing | PyMuPDF / pypdf |
| Embeddings | Sentence-Transformers |
| Embedding model | `all-MiniLM-L6-v2` |
| Vector search | FAISS |
| LLM runtime | Ollama |
| Local LLM | `llama3.2:3b` |
| Numerical reasoning | Python |

The project runs locally and does **not require a paid API or API key**.

## Project Structure

```text
policysense/
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── llm.py
│   ├── policy_analyzer.py
│   └── comparison.py
│
└── sample_policies/
    ├── sample_attendance_policy.pdf
    └── sample_attendance_policy_revised.pdf
```

### Module responsibilities

- `app.py` — Streamlit interface and application flow
- `pdf_processor.py` — extracts PDF text and creates chunks
- `embeddings.py` — creates local semantic embeddings
- `retriever.py` — builds and searches the FAISS vector index
- `llm.py` — communicates with the local Ollama model
- `policy_analyzer.py` — coordinates retrieval, LLM processing, and calculations
- `comparison.py` — handles policy version comparison

## Getting Started

### Requirements

- Python 3.10+
- Ollama
- A computer capable of running the selected local LLM

### 1. Install Ollama

Download Ollama for Windows:

https://ollama.com/download

After installation, Ollama runs as a local service.

### 2. Download the LLM

Open PowerShell or Command Prompt:

```bash
ollama pull llama3.2:3b
```

Verify:

```bash
ollama list
```

The output should include:

```text
llama3.2:3b
```

### 3. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 4. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

The first installation may be relatively large because Sentence-Transformers uses PyTorch.

### 6. Run the application

```bash
streamlit run app.py
```

The application normally opens at:

```text
http://localhost:8501
```

Ollama must be running in the background while the application is being used.

## Sample Questions

The repository includes sample attendance/scholarship policies that can be used to test the application.

Try:

1. `I attended 32 out of 45 classes. What happens if I miss 3 more?`
2. `Am I eligible for the Merit Scholarship if my CGPA is 8.2?`
3. `What documents do I need to apply for the scholarship?`
4. `What happens if I miss the scholarship deadline?`
5. `If my attendance is 68%, can I still sit for the exam?`

## Why RAG?

Policy documents can contain many sections, clauses, conditions, and exceptions. Instead of passing the entire document to the LLM for every question, PolicySense:

1. Converts policy chunks into embeddings.
2. Embeds the user's question.
3. Uses FAISS to retrieve the most relevant chunks.
4. Provides those chunks to the LLM as context.
5. Generates an answer grounded in the retrieved policy content.

This makes the response more focused and allows the application to show the evidence used.

## Grounding and Numerical Reasoning

The system is designed to reduce unsupported responses by instructing the local LLM to rely on retrieved policy excerpts.

For supported numerical scenarios, Python performs calculations rather than relying on the LLM to do arithmetic.

For example:

```text
Attendance = classes attended / total classes × 100
```

This separates deterministic computation from language generation.

## Limitations

- The local `llama3.2:3b` model is smaller and less capable than larger hosted models.
- Retrieval may miss information when an answer depends on multiple distant sections.
- PDF extraction can be imperfect for scanned or unusually formatted documents.
- The current calculator supports specific policy scenarios rather than arbitrary mathematical reasoning.
- Policy comparison works at the document/paragraph level and may be less reliable when documents are heavily reformatted.
- Grounding reduces hallucination risk but does not eliminate it.

This is an educational portfolio project and should not be treated as a final legal, financial, medical, or administrative decision-maker.

## Future Improvements

Potential extensions include:

- Build an evaluation dataset to measure retrieval accuracy and answer faithfulness.
- Experiment with hybrid keyword + vector retrieval.
- Add a re-ranking stage for retrieved policy chunks.
- Improve chunking using policy section and clause structure.
- Generalize the calculation component using tool calling.
- Compare different local LLMs and retrieval strategies quantitatively.

## License

This project is intended for educational and portfolio use.
