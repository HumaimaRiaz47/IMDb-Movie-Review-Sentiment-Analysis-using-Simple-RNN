# 🎬 IMDb Movie Review Sentiment Analysis using Simple RNN

## 📌 Overview

This project performs **binary sentiment classification** on movie reviews using a **Simple Recurrent Neural Network (Simple RNN)** built with TensorFlow/Keras.

The model predicts whether a movie review expresses a **Positive** or **Negative** sentiment.

A **Streamlit** web application is included for real-time predictions.

---

## 🚀 Features

* IMDb Movie Review Sentiment Analysis
* Text preprocessing using TextVectorization
* Simple RNN Neural Network
* Streamlit Web Application
* Model saved in Keras format
* Vocabulary saved separately
* Real-time sentiment prediction

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* Streamlit
* Pandas
* NumPy
* Scikit-learn

---

## 📂 Project Structure

```
IMDb-Sentiment-Analysis/
│
├── IMDB Dataset.csv
├── train.py
├── predict.py
├── app.py
├── imdb_model.keras
├── vocab.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

IMDb Dataset of 50K Movie Reviews

* 50,000 movie reviews
* Binary Sentiment Classification
* Labels:

  * Positive
  * Negative

---

## 🧠 Model Architecture

```
Input Review
      │
      ▼
Text Vectorization
      │
      ▼
Embedding Layer
      │
      ▼
Simple RNN
      │
      ▼
Dense Layer (Sigmoid)
      │
      ▼
Sentiment Prediction
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/IMDb-Sentiment-Analysis.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

This generates:

* imdb_model.keras
* vocab.pkl

---

## ▶️ Predict Using Python

```bash
python predict.py
```

---

## ▶️ Run Streamlit

```bash
streamlit run app.py
```

---

## 📈 Sample Output

**Review**

> This movie was fantastic! The acting was amazing.

Prediction

```
Positive 😊
Score: 0.97
```

---

## Demo

<video controls src="20260616-1911-14.6500764.mp4" title="Title"></video>

## Author

**Humaima Riaz**

Computer Systems Engineering Student

Interested in:

* Artificial Intelligence
* Deep Learning
* Natural Language Processing
* Machine Learning

---

⭐ If you found this project useful, don't forget to star the repository!
