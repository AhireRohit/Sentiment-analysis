# 🎬 IMDb Sentiment Analysis using DistilBERT  
Fine-tuned Transformer model that classifies IMDb movie reviews as **Positive** or **Negative**.  
Built using **HuggingFace Transformers**, **PyTorch**, and an interactive **Streamlit web app**.

# ⭐ Project Overview  
This project implements a complete sentiment analysis pipeline:

- ✔ IMDb dataset (50,000 labeled movie reviews)  
- ✔ Text preprocessing (cleaning, tokenization)  
- ✔ Fine-tuning **DistilBERT** for classification  
- ✔ GPU-accelerated training  
- ✔ High-accuracy predictions  
- ✔ Beautiful **Streamlit UI**  
- ✔ Exportable + reusable model  


# 🚀 Features  
- 🌟 **DistilBERT** fine-tuned on IMDb  
- 💬 Real-time text classification  
- 🎨 Beautiful Streamlit UI  
- 📊 Confidence & probability visualization  
- 🔥 Softmax prediction bars  
- ⚡ Fast CUDA-accelerated inference  
- 🧹 Clean preprocessing pipeline  

---

# 📦 Installation


pip install transformers datasets torch streamlit scikit-learn

# 🧠 Model Training
1️⃣ Load IMDb Dataset
from datasets import load_dataset
dataset = load_dataset("imdb")

2️⃣ Preprocess & Tokenize

Tokenization using DistilBERT tokenizer.

3️⃣ Train
trainer.train()

4️⃣ Save Model
trainer.save_model("distilbert_imdb_model")

# 📊 Model Performance
Metric	Score
Training Loss	~0.18
Validation Loss	~0.26
Accuracy	~93–95%
GPU	NVIDIA RTX 3050 (Laptop)
🧪 Inference Example
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert_imdb_model")
model = AutoModelForSequenceClassification.from_pretrained("distilbert_imdb_model")

text = "This movie was absolutely amazing!"

inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)

pred = torch.argmax(outputs.logits).item()
print("Positive" if pred == 1 else "Negative")

# 🌐 Streamlit App
Run locally:
streamlit run app.py

Features:

Text box input

Sentiment label

Confidence percentage

Probability bars

Clean UI

# 📁 Folder Structure
📦 imdb-sentiment-analysis
│
├── distilbert_imdb_model/
│   ├── config.json
│   ├── pytorch_model.bin
│   ├── tokenizer.json
│   ├── special_tokens_map.json
│   ├── vocab.txt
│
├── app.py
├── train_distilbert.ipynb
├── README.md
└── requirements.txt