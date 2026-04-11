from __future__ import annotations

import streamlit as st

from src.preprocess import LABEL_MAP, load_artifacts, predict_text, train_and_save_artifacts


st.set_page_config(page_title="Hate Speech Detection", layout="centered")

st.title("Hate Speech Detection")
st.write(
    "Classify input text into `hate_speech`, `offensive_language`, or `neither` "
    "using the final TF-IDF + LinearSVC submission pipeline."
)

with st.sidebar:
    st.header("Model")
    retrain = st.button("Retrain model")

if retrain:
    vectorizer, model = train_and_save_artifacts()
    st.sidebar.success("Model retrained and saved.")
else:
    vectorizer, model = load_artifacts()

user_text = st.text_area("Enter text", height=140, placeholder="Type a tweet or sentence...")

if st.button("Predict", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text before predicting.")
    else:
        result = predict_text(model, vectorizer, user_text)
        st.subheader("Prediction")
        st.write(f"Label: `{result['label']}`")
        st.write(f"Class ID: `{result['class_id']}`")
        st.write(f"Confidence: `{result['confidence']:.4f}`")
        st.write(f"Cleaned text: `{result['cleaned_text']}`")

        score_rows = [
            {"class_id": class_id, "label": LABEL_MAP[class_id], "score": float(score)}
            for class_id, score in enumerate(result["scores"])
        ]
        st.dataframe(score_rows, use_container_width=True)
