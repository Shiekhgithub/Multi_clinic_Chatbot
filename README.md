# 🏥 Healthcare RAG System

A **multi-dataset Retrieval-Augmented Generation (RAG)** system built with LangChain, ChromaDB, and Streamlit. Supports intelligent conversational querying over three healthcare datasets.

---

## 📁 Project Structure

```
healthcare_rag/
├── data/                      ← Place your CSV files here
│   ├── heart.csv
│   ├── dermatology.csv
│   └── diabetes.csv
├── chroma_db/                 ← Auto-created: persisted vector stores
├── data_ingestion.py          ← CSV loading, cleaning, row→text conversion
├── vector_store.py            ← ChromaDB collections builder/loader
├── agent.py                   ← Tools + ReAct agent setup
├── ingest.py                  ← One-time ingestion script
├── app.py                     ← Streamlit web UI
├── cli_chat.py                ← Terminal chat interface
├── requirements.txt
├── .env.example               ← Copy to .env and fill in your API key
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / copy files
```bash
git clone https://github.com/javaidgithub/healthcare-rag-chatbot.git
cd healthcare-rag-chatbot
python -m venv venv
venv\Scripts\activate   # (Windows)
pip install -r requirements.txt
```

OR
Place all project files in a folder, e.g. `healthcare_rag/`.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If you are on Windows and Streamlit later reports `ModuleNotFoundError: No module named 'torchvision'`, install the CPU wheels explicitly:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 3. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and add your API key:

```env
# Free option — get a key at https://console.groq.com
GROQ_API_KEY=your_groq_api_key_here
LLM_PROVIDER=groq
```

### 4. Download datasets from Kaggle

| Dataset | Kaggle URL | Save as |
|---------|-----------|---------|
| Heart Disease | [link](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) | `data/heart.csv` |
| Dermatology | [link](https://www.kaggle.com/datasets/olcaybolat1/dermatology-dataset-classification) | `data/dermatology.csv` |
| Pakistani Diabetes | [link](https://www.kaggle.com/datasets/mshoaibishaaq/pakistani-diabetes-dataset) | `data/diabetes.csv` |

Create the `data/` folder and place the CSVs inside.

### 5. Build vector stores (run once)

```bash
python ingest.py
```

This reads the CSVs, converts each row to natural-language text, generates embeddings, and saves three ChromaDB collections to `./chroma_db/`.

### 6. Launch the chatbot

**Streamlit (recommended):**
```bash
streamlit run app.py
```

**Terminal CLI:**
```bash
python cli_chat.py
```

---

## 🏗️ Architecture

```
User Query
    │
    ▼
ReAct Agent (LangChain)
    │
    ├─ Decides which Tool(s) to use
    │
    ├── heart_disease_tool ──► heart_disease_index (ChromaDB)
    ├── dermatology_tool   ──► dermatology_index   (ChromaDB)
    └── diabetes_tool      ──► diabetes_index      (ChromaDB)
                                      │
                              HuggingFace Embeddings
                         (sentence-transformers/all-MiniLM-L6-v2)
                                      │
                                 Retrieved Docs
                                      │
                                    LLM
                                (Groq / OpenAI)
                                      │
                                   Answer
```

---

## 🔧 Key Components

### Data Ingestion (`data_ingestion.py`)
- Loads CSVs with pandas
- Normalises column names (lowercase, underscores)
- Drops columns with >50% missing values
- Fills remaining NaNs (median for numerics, "unknown" for text)
- Converts each row → structured natural-language sentence

### Vector Store (`vector_store.py`)
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (free, local)
- 3 separate ChromaDB collections:
  - `heart_disease_index`
  - `dermatology_index`
  - `diabetes_index`
- Persistent local storage

### Agent (`agent.py`)
- Three `create_retriever_tool` tools (one per dataset)
- ReAct agent using `hwchase17/react-chat` prompt from LangChain Hub
- `ConversationBufferMemory` for multi-turn context
- Configurable `k` (documents retrieved per query)

---

## 💡 Example Questions

- *"What are common features of patients with heart disease?"*
- *"Tell me about psoriasis cases in the dermatology dataset."*
- *"What is the average glucose level in diabetic patients?"*
- *"Do older patients tend to have higher cholesterol?"*
- *"What skin conditions involve erythema?"*

---

## ⚙️ LLM Providers

| Provider | Setup | Notes |
|----------|-------|-------|
| **Groq** | Free at [console.groq.com](https://console.groq.com) | Fast, recommended |
| **OpenAI** | [platform.openai.com](https://platform.openai.com) | Paid |
| **Gemini** | [aistudio.google.com](https://aistudio.google.com/apikey) | Free tier may be quota-limited by region/project |

Switch provider in `.env`:
```env
LLM_PROVIDER=groq   # or openai or gemini
```

---

## 🔍 Streamlit Features

- 💬 Multi-turn conversation with memory
- 🔍 Optional reasoning trace (shows which tool was used and why)
- 💡 Example questions panel
- ⚙️ Sidebar controls for `k` and reasoning trace toggle
- 🗑️ Clear conversation button

---

## 📝 Notes

- Vector stores are built **once** and reused on every run.
- To **rebuild** stores (after new data), delete `./chroma_db/` and re-run `python ingest.py`.
- The embedding model runs **locally** on CPU — no API key needed for embeddings.
- The agent automatically routes questions to the correct dataset(s).
- `streamlit` plus recent `transformers` builds can touch optional vision modules during file watching. This repo now includes `torchvision` explicitly to avoid that startup error.
