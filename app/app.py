import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Sentiment Dashboard", layout="wide")

# ------------------ CUSTOM UI ------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3 {
    text-align: center;
}

.stButton>button {
    background-color: #ff4b2b;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
}

.stTextArea textarea {
    border-radius: 10px;
    font-size: 16px;
}

.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# ------------------ TITLE ------------------
st.title("🚀 AI Sentiment Analysis Dashboard")
st.write("Analyze emotions from text with AI")

# ------------------ SIDEBAR ------------------
option = st.sidebar.radio("Choose Input Type", ["Single Text", "Upload CSV"])

# ------------------ PREDICT FUNCTION ------------------
def predict(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec).max()
    return pred, prob

# ------------------ SINGLE TEXT ------------------
if option == "Single Text":
    st.subheader("💬 Enter your text")

    text = st.text_area("Type your sentence here...")

    if st.button("Analyze"):
        if text.strip() != "":
            pred, prob = predict(text)

            emoji = "😄" if pred == "positive" else "😡" if pred == "negative" else "😐"
            color = "green" if pred == "positive" else "red" if pred == "negative" else "orange"

            # Result Card
            st.markdown(f"""
            <div style='padding:25px; border-radius:15px; background-color:#1f1f2e; text-align:center;'>
                <h2 style='color:{color}'>{emoji} {pred.upper()}</h2>
                <p>Confidence: {round(prob*100,2)}%</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(prob * 100))

        else:
            st.warning("Please enter some text")

# ------------------ CSV UPLOAD ------------------
else:
    st.subheader("📁 Upload your CSV file")

    file = st.file_uploader("Upload CSV", type=["csv"])

    if file is not None:
        data = pd.read_csv(file)

        if "text" not in data.columns:
            st.error("CSV must contain a column named 'text'")
        else:
            st.success("File uploaded successfully!")

            # Predictions
            data["Prediction"] = model.predict(vectorizer.transform(data["text"]))

            st.subheader("📊 Data Preview")
            st.dataframe(data.head())

            # Pie Chart
            fig1 = px.pie(data, names="Prediction", title="Sentiment Distribution")
            st.plotly_chart(fig1, use_container_width=True)

            # Bar Chart
            counts = data["Prediction"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]

            fig2 = px.bar(counts, x="Sentiment", y="Count", title="Sentiment Count")
            st.plotly_chart(fig2, use_container_width=True)

            # Word Cloud
            st.subheader("☁️ Word Cloud")

            text_all = " ".join(data["text"].astype(str))
            wc = WordCloud(width=800, height=400).generate(text_all)

            fig, ax = plt.subplots()
            ax.imshow(wc)
            ax.axis("off")

            st.pyplot(fig)