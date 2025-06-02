# Efficient Intent Detection via Vector Retrieval

This project presents a lightweight and scalable approach to intent detection, reframing the task as a **vector similarity search** rather than a supervised classification problem. The method is based on *sentence embeddings*, *FAISS indexing*, and *majority voting* over the k-nearest neighbors, enabling zero-shot inference with support for **out-of-scope (OOS)** rejection.

---

## 🚀 Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook efficient-intent-detection-via-vector-retrieval.ipynb
```

---

## 📊 Overview

### Key Features

* **No fine-tuning required**: Uses off-the-shelf `sentence-transformers`
* **Fast and efficient**: Sub-20ms inference with FAISS HNSW indexing
* **OOS detection**: A simple similarity threshold `τ` handles unknown queries
* **Extensible**: Add new intents without retraining

### Pipeline

1. Encode training examples using a pre-trained bi-encoder (e.g., MPNet, BGE).
2. Build a FAISS index using HNSW for fast k-NN search.
3. For each user query:

   * Encode the input
   * Retrieve `k` most similar examples
   * Predict by majority vote
   * Reject as OOS if the top similarity < threshold `τ`

---

## 🧠 Methodology

### Dataset

We use [Clinc-OOS Plus](https://huggingface.co/datasets/clinc_oos), with:

* 150 **in-scope** intents
* 1 **out-of-scope (OOS)** class
* \~22k utterances split across training, validation, and test

### Models Evaluated

* `bge-large-en-v1.5`
* `all-mpnet-base-v2`
* `gtr-t5-large`
* `all-MiniLM-L6-v2`, etc.

### Configuration Example

The best configuration achieved:

* **Top-1 Accuracy**: 89.42%
* **Macro-F1**: 90.98%
* **OOS Recall**: 76.6%
* **Latency**: \~21 ms (on a T4 GPU)

```python
model_name = "bge-large-en-v1.5"
k = 5
tau = 0.75
```

---

## 📊 Inference Logic

```python
def predict_intent(query: str, index, model, k: int, tau: float) -> str:
    embedding = model.encode(query)
    D, I = index.search([embedding], k)
    intents = [labels[i] for i in I[0]]

    # Majority voting
    counter = Counter(intents)
    top_intents = counter.most_common()

    # Tie-break using average distance
    if len(top_intents) > 1 and top_intents[0][1] == top_intents[1][1]:
        intent_dist = defaultdict(list)
        for label, dist in zip(intents, D[0]):
            intent_dist[label].append(dist)
        chosen = min(intent_dist.items(), key=lambda x: np.mean(x[1]))[0]
    else:
        chosen = top_intents[0][0]

    # OOS rejection
    if D[0][0] < tau:
        return "OOS"
    return chosen
```

---

## 📈 Evaluation Metrics

| Metric     | Description                            |
| ---------- | -------------------------------------- |
| Accuracy   | Correct Top-1 predictions              |
| Macro-F1   | Mean F1 score across all classes       |
| Recall-OOS | True positives among OOS samples       |
| Latency    | Mean end-to-end inference time (in ms) |

---

## 🔄 Updating the Index

To add a new intent:

```python
new_sentences = ["Book a surf lesson in Hawaii", "Schedule scuba diving"]
new_vectors = model.encode(new_sentences)
index.add(np.array(new_vectors))
labels.extend(["surf_lesson", "scuba_dive"])
```

> 🧹 No retraining required. Simply encode and insert.

---

## 🗝️ Future Work

* Class-specific thresholds `τ_c` for nuanced OOS detection
* Multilingual support (e.g., using `LaBSE`)
* Lightweight cross-encoder re-ranking after k-NN
* Energy/memory benchmarks at scale

---

## 📚 References

1. Cirolini, R. & Rigo, S. (2024). *Efficient Intent Detection via Vector Retrieval*. UNISINOS.
2. Larson, S. et al. (2019). *Clinc-OOS Benchmark*. EMNLP.
3. Reimers, N. & Gurevych, I. (2019). *Sentence-BERT*.

---

## 🤝 Contributions

Feel free to fork this repo or adapt the notebook. For questions or collaboration, contact the authors via the paper or open a discussion.

---

## 📄 License

MIT License
