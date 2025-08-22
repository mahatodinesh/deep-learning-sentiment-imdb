#  IMDb Sentiment Analysis using LSTM  

This project demonstrates **sentiment analysis** on the IMDb 50K movie reviews dataset using a **deep learning LSTM model**. The model classifies reviews as **positive** or **negative**.  

---

##  Project Workflow  

### 1. Import Dependencies  
- Python packages: `os`, `json`, `pandas`, `scikit-learn`, `tensorflow.keras`  
- Kaggle API for dataset download  

### 2. Data Collection (Kaggle API)  
- Download dataset: [IMDb 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)  
- Extract and load `IMDB Dataset.csv`  

### 3. Data Exploration  
- 50,000 reviews (25k positive, 25k negative)  
- Balanced dataset  

### 4. Data Preprocessing  
- Convert labels: `positive → 1`, `negative → 0`  
- Split into train (40k) and test (10k) sets  
- Tokenize reviews (top 5000 words)  
- Pad sequences to uniform length (200 tokens)  

### 5. Model Architecture (LSTM)  
```text
Embedding(5000 → 128) → LSTM(128, dropout=0.2) → Dense(1, sigmoid)
```
- **Embedding Layer**: Converts words into dense vectors  
- **LSTM Layer (128 units)**: Captures sequential context  
- **Dense Layer**: Outputs sentiment (positive/negative)  

### 6. Training  
- Optimizer: `Adam`  
- Loss: `Binary Crossentropy`  
- Metrics: `Accuracy`  
- Epochs: `5`  
- Batch size: `64`  
- Validation split: `0.2`  

**Training Results:**  
- Final Training Accuracy: ~93%  
- Validation Accuracy: ~87%  

### 7. Model Evaluation  
```text
Test Loss: 0.3138
Test Accuracy: 0.8830
```
➡ The model generalizes well with **88% accuracy** on unseen test data.  

### 8. Predictive System  
Function to classify new reviews:  

```python
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment
```

**Examples:**  
- `"This movie was fantastic. I loved it."` →  positive  
- `"This movie was not that good"` →  negative  
- `"This movie was ok but not that good."` →  negative  

---

##  Key Results  
- Balanced IMDb dataset (50K reviews)  
- Achieved **88% accuracy** on test set  
- Works well for short text reviews  

---

##  Future Enhancements  
- Use **Bidirectional LSTMs / GRU** for better accuracy  
- Incorporate **pretrained embeddings** (GloVe, Word2Vec)  
- Deploy model with **Flask / Streamlit** for interactive use  

---

##  Requirements  
- Python 3.10+  
- TensorFlow  
- Pandas  
- Scikit-learn  
- Kaggle API  

Install with:  
```bash
pip install kaggle pandas scikit-learn tensorflow
```

---

## 👨‍💻 Author  
Built with ❤️ using **Python, TensorFlow, and Kaggle API**.  
