# CSL 7640 — Natural Language Understanding
## Assignment 2

> **Course:** CSL 7640: Natural Language Understanding  
> **Instructor:** Anand Mishra  
> **Institute:** Indian Institute of Technology Jodhpur  

---

## 📁 Repository Structure

```
├── Problem1_Word2Vec/
│   └── Problem1_Word2Vec.ipynb       # Word embeddings from IIT Jodhpur data
│
├── Problem2_NameGeneration/
│   └── Problem2_NameGeneration.ipynb # Character-level name generation using RNN variants
│
└── README.md
```

---

## ⚙️ Environment

Both notebooks are designed to run on **Google Colab** (free tier is sufficient).  
No local setup is required. A GPU runtime is optional but recommended for Problem 2.

> **Recommended runtime:** `Runtime → Change runtime type → T4 GPU`  
> Problem 1 (Word2Vec) works fine on CPU. Problem 2 (RNN training) benefits from GPU.

---

## 🔵 Problem 1 — Learning Word Embeddings from IIT Jodhpur Data

**Notebook:** `Problem1_Word2Vec/Problem1_Word2Vec.ipynb`

### What it does
- Scrapes textual data from IIT Jodhpur official website pages (departments, academics, research, admissions, campus, people)
- Optionally ingests IIT Jodhpur PDF documents (academic regulations, course syllabi, etc.)
- Trains **Word2Vec from scratch** (NumPy only) for both CBOW and Skip-gram with Negative Sampling
- Trains equivalent **Gensim Word2Vec** models for comparison
- Evaluates embeddings via cosine similarity, nearest neighbors, and analogy experiments
- Visualizes embeddings using PCA and t-SNE

### How to run

**Step 1 — Open in Colab**  
Upload `Problem1_Word2Vec.ipynb` to [colab.research.google.com](https://colab.research.google.com) or open directly from GitHub via the Colab badge.

**Step 2 — Run blocks in order**

| Block | Description |
|-------|-------------|
| Block 1 | Install dependencies (`gensim`, `nltk`, `beautifulsoup4`, `wordcloud`, etc.) |
| Block 2 | Import all libraries |
| Block 3 | Define scraper utility (SSL bypass for IIT Jodhpur subdomains) |
| Block 4 | Define IIT Jodhpur URLs to scrape |
| Block 5 | Scrape websites + add fallback domain text |
| Block 5B | *(Optional)* Upload IIT Jodhpur PDF files for richer corpus |
| Block 6 | Preprocess corpus — tokenize, lowercase, remove noise |
| Block 7 | Generate Word Cloud |
| Block 8 | Define `Word2VecScratch` class (CBOW + Skip-gram from scratch) |
| Block 9 | Train all 4 scratch models (SG-A, SG-B, CBOW-A, CBOW-B) |
| Block 10 | Train Gensim models for comparison |
| Block 11 | Semantic analysis — top-5 nearest neighbors |
| Block 12 | Analogy experiments (data-driven, auto-selects from actual vocab) |
| Block 13 | PCA visualizations |
| Block 14 | t-SNE visualizations |
| Block 15 | Hyperparameter comparison table |
| Block 16 | Save all output files |
| Block 17 | Zip and download all outputs |

> ⚠️ **Block 5 (scraping) takes 3–5 minutes.** IIT Jodhpur uses JavaScript-heavy pages — some URLs may return limited content. The fallback corpus ensures minimum viable training data.

> 📄 **Block 5B (PDF upload) is optional but recommended.** Upload any IIT Jodhpur PDFs (academic regulations document, BTech/MTech/PhD regulations, course syllabi). The file picker will open automatically when the block runs.

### Outputs generated
After running all blocks, the following files are saved and zipped for download:

| File | Description |
|------|-------------|
| `cleaned_corpus.txt` | One preprocessed sentence per line |
| `raw_corpus.json` | Raw scraped text per URL |
| `processed_corpus.pkl` | Tokenized sentences (Python list) |
| `scratch_skipgram_A.pkl` | Scratch Skip-gram model A checkpoint |
| `scratch_skipgram_B.pkl` | Scratch Skip-gram model B checkpoint |
| `scratch_cbow_A.pkl` | Scratch CBOW model A checkpoint |
| `scratch_cbow_B.pkl` | Scratch CBOW model B checkpoint |
| `gensim_skipgram.model` | Gensim Skip-gram model |
| `gensim_cbow.model` | Gensim CBOW model |
| `wordcloud.png` | Word Cloud visualization |
| `pca_*.png` (×4) | PCA 2D projections for each model |
| `tsne_*.png` (×4) | t-SNE 2D projections for each model |

### Key hyperparameters

| Model | Type | Dim | Window | Neg Samples | Epochs |
|-------|------|-----|--------|-------------|--------|
| Scratch SG-A | Skip-gram | 100 | 5 | 5 | 15 |
| Scratch SG-B | Skip-gram | 50 | 3 | 10 | 15 |
| Scratch CBOW-A | CBOW | 100 | 5 | 5 | 15 |
| Scratch CBOW-B | CBOW | 50 | 3 | 10 | 15 |
| Gensim SG | Skip-gram | 100 | 5 | 5 | 5 |
| Gensim CBOW | CBOW | 100 | 5 | 5 | 5 |

---

## 🟠 Problem 2 — Character-Level Name Generation using RNN Variants

**Notebook:** `Problem2_NameGeneration/Problem2_NameGeneration.ipynb`

### What it does
- Uses a dataset of 1000 Indian names (`TrainingNames.txt`) generated using an LLM
- Implements and trains three sequence models **from scratch**:
  - Vanilla RNN
  - Bidirectional LSTM (BLSTM)
  - RNN with Basic Attention Mechanism
- Evaluates each model on **Novelty Rate** and **Diversity**
- Generates representative name samples and reports qualitative analysis

### How to run

**Step 1 — Enable GPU (recommended)**  
`Runtime → Change runtime type → T4 GPU`  
Training RNN/LSTM models is significantly faster on GPU.

**Step 2 — Open in Colab**  
Upload `Problem2_NameGeneration/Problem2_NameGeneration.ipynb` to Colab.

**Step 3 — Run blocks in order**

| Block | Description |
|-------|-------------|
| Block 1 | Install dependencies |
| Block 2 | Imports and device setup (auto-detects GPU/CPU) |
| Block 3 | Generate / load `TrainingNames.txt` (1000 Indian names) |
| Block 4 | Dataset preprocessing — character vocabulary, one-hot encoding |
| Block 5 | Vanilla RNN implementation from scratch |
| Block 6 | Bidirectional LSTM (BLSTM) implementation from scratch |
| Block 7 | RNN with Attention Mechanism implementation from scratch |
| Block 8 | Train all three models |
| Block 9 | Generate names from each model |
| Block 10 | Compute Novelty Rate and Diversity metrics |
| Block 11 | Qualitative analysis — sample outputs and failure modes |
| Block 12 | Save outputs and download |

> ⚠️ **`TrainingNames.txt` must exist before Block 4.** Block 3 generates it automatically. If you want to use your own name list, upload a plain text file with one name per line (1000 names) and skip Block 3.

### Outputs generated

| File | Description |
|------|-------------|
| `TrainingNames.txt` | 1000 Indian training names (one per line) |
| `generated_rnn.txt` | Names generated by Vanilla RNN |
| `generated_blstm.txt` | Names generated by BLSTM |
| `generated_attention.txt` | Names generated by RNN+Attention |
| `evaluation_results.csv` | Novelty Rate and Diversity scores per model |
| `model_rnn.pt` | Saved Vanilla RNN weights |
| `model_blstm.pt` | Saved BLSTM weights |
| `model_attention.pt` | Saved RNN+Attention weights |

### Model architectures summary

| Model | Architecture | Trainable Params (approx.) |
|-------|-------------|---------------------------|
| Vanilla RNN | Single-layer RNN, character embeddings → hidden → softmax | ~50K |
| BLSTM | Bidirectional LSTM, forward + backward hidden states concatenated | ~200K |
| RNN + Attention | RNN encoder with additive attention over hidden states | ~80K |

> Exact parameter counts are printed during Block 8 (training).

---

## 📦 Dependencies

All dependencies are installed automatically in Block 1 of each notebook. For reference:

**Problem 1:**
```
gensim, nltk, beautifulsoup4, requests, wordcloud,
matplotlib, scikit-learn, numpy, tqdm, pymupdf
```

**Problem 2:**
```
torch, numpy, matplotlib, pandas, tqdm
```

---

## 📋 Deliverables Checklist

### Problem 1
- [x] Source code (well-documented, block-by-block)
- [x] Cleaned corpus file (`cleaned_corpus.txt`)
- [x] Visualizations (Word Cloud, PCA ×4, t-SNE ×4)
- [x] Report (separate PDF/DOCX)

### Problem 2
- [x] Source code for all three models
- [x] Generated name samples (`generated_*.txt`)
- [x] Evaluation scripts (Novelty Rate, Diversity — built into notebook)
- [x] Report (separate PDF/DOCX)

---
