# IMDb Movie Review Sentiment Analysis using Simple RNN

## 📌 Project Overview

This project implements an end-to-end Deep Learning solution for sentiment analysis on movie reviews using a Simple Recurrent Neural Network (RNN). The model is trained on the IMDb Movie Reviews Dataset to classify reviews as either **Positive** or **Negative**.

The project covers the complete Deep Learning workflow, including:

* Data preprocessing and feature engineering
* Text tokenization and sequence padding
* Word Embedding Layer
* Simple RNN model training
* Model evaluation
* Streamlit web application
* Deployment-ready pipeline

---

## 🎯 Problem Statement

Movie reviews contain valuable information about audience opinions. Manually analyzing thousands of reviews is time-consuming and inefficient.

The objective of this project is to build a Deep Learning model capable of automatically determining whether a movie review expresses a positive or negative sentiment.

---

## 📂 Dataset

**Dataset:** IMDb Movie Reviews Dataset

* Total Reviews: 50,000
* Binary Sentiment Classification
* Labels:

  * Positive (1)
  * Negative (0)

---

## 🏗️ Project Architecture

```text
Input Review Text
       │
       ▼
Text Preprocessing
       │
       ▼
Tokenization & Padding
       │
       ▼
Embedding Layer
       │
       ▼
Simple RNN Layer
       │
       ▼
Dense Layer
       │
       ▼
Sigmoid Activation
       │
       ▼
Sentiment Prediction
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-Learn
* Streamlit

---

## 🧠 Model Architecture

```python
model = Sequential([
    Embedding(input_dim=vocab_size,
              output_dim=128,
              input_length=max_length),

    SimpleRNN(128),

    Dense(1, activation='sigmoid')
])
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/HumaimaRiaz47/IMDb-Movie-Review-Sentiment-Analysis-using-Simple-RNN.git
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the Model

```bash
python train.py
```

### 4. Run the Streamlit Application

```bash
streamlit run app.py
```

---

## 📊 Sample Predictions

### Positive Review

**Input**

```text
This movie was absolutely amazing and the acting was brilliant.
```

**Prediction**

```text
Positive Sentiment 😊
```

### Negative Review

**Input**

```text
This was the worst movie I have ever watched.
```

**Prediction**

```text
Negative Sentiment 😞
```

---

## 🔮 Future Improvements

* LSTM (Long Short-Term Memory)
* GRU (Gated Recurrent Unit)
* Bidirectional RNN
* Attention Mechanisms
* Transformer Models (BERT)
* Hyperparameter Optimization

---

## 📚 Learning Outcomes

This project helped in understanding:

* Natural Language Processing (NLP)
* Word Embeddings
* Sequence Modeling
* Recurrent Neural Networks (RNNs)
* Deep Learning Model Deployment
* Streamlit Application Development

---

## 👩‍💻 Author

**Humaima Riaz**

GitHub: https://github.com/HumaimaRiaz47

---

⭐ If you found this project useful, consider giving it a star on GitHub!
