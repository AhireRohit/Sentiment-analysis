import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model
tokenizer = AutoTokenizer.from_pretrained("distilbert_imdb_model")
model = AutoModelForSequenceClassification.from_pretrained("distilbert_imdb_model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# --------- Prediction Function ---------
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    probs = F.softmax(logits, dim=1).detach().cpu().numpy()[0]
    
    pred_class = probs.argmax()
    confidence = probs[pred_class]
    
    return pred_class, confidence, probs


# --------- Streamlit UI ---------
st.set_page_config(
    page_title="IMDb Sentiment Analyzer",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown(
    """
    <h1 style='text-align: center; color: #ff4b4b;'>🎬 IMDb Sentiment Analyzer</h1>
    <p style='text-align: center; font-size: 18px;'>Analyze movie reviews using <b>DistilBERT</b> fine-tuned on the IMDb dataset.</p>
    """,
    unsafe_allow_html=True
)

st.write("")

# Text Input Box
review = st.text_area(
    "📝 Enter your movie review here:",
    height=180,
    placeholder="Type a movie review... (e.g., 'This movie was absolutely amazing!')"
)

if st.button("🔍 Analyze Sentiment", use_container_width=True):
    
    if review.strip() == "":
        st.warning("Please enter a review text.")
    else:
        with st.spinner("Analyzing with DistilBERT... 🔄"):
            pred_class, confidence, probs = predict(review)
        
        # Label Mapping
        sentiment_label = "Positive 😊" if pred_class == 1 else "Negative 😡"
        sentiment_color = "#2ecc71" if pred_class == 1 else "#e74c3c"

        # Result Card
        st.markdown(
            f"""
            <div style='padding: 20px; border-radius: 10px; background-color: {sentiment_color}; color: white; text-align: center;'>
                <h2>Sentiment: {sentiment_label}</h2>
                <h3>Confidence: {confidence*100:.2f}%</h3>
            </div>
            """, 
            unsafe_allow_html=True
        )

        st.write("")
        st.subheader("📊 Probability Distribution")
        
        # Probability bars
        st.progress(float(probs[1]))  # positive
        st.write(f"**Positive:** {probs[1]*100:.2f}%")
        
        st.progress(float(probs[0]))  # negative
        st.write(f"**Negative:** {probs[0]*100:.2f}%")

        st.write("")
        st.info("Tip: Try entering different types of reviews — emotional, sarcastic, or mixed opinions!")

# Footer
st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px;'>
        Built with ❤️ using <b>DistilBERT</b> and <b>Streamlit</b>.
    </p>
    """,
    unsafe_allow_html=True
)
