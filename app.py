import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
st.set_page_config(
    page_title="Medical AI Chatbot",
    page_icon="🧠",
    layout="wide"
)
st.markdown("""
<style>
.main {
    background-color: #0E1117;
    color: white;
}
.chat-user {
    background-color: #2b313e;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 10px;
    color: white;
}
.chat-bot {
    background-color: #444654;
    padding: 15px;
    border-radius: 12px;
    margin-bottom: 20px;
    color: white;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #00FFAA;
}
.subtitle {
    text-align: center;
    color: #BBBBBB;
    margin-bottom: 30px;
}
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.title("🧠 About")
    st.write("""
    This AI chatbot provides educational information about:
    ✅ Noonan Syndrome  
    ✅ Genetic Disorders  
    ✅ Symptoms  
    ✅ Diagnosis  
    ✅ Treatment  
    ✅ Inheritance  
    """)
    st.warning(
        "Educational purposes only. Not medical advice."
    )
st.markdown(
    '<div class="title">Medical AI Chatbot</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">Powered by Sentence Transformers</div>',
    unsafe_allow_html=True
)
@st.cache_data
def load_data():
    data = pd.read_csv("medical_dataset.csv")
    return data
data = load_data()
questions = data["question"].tolist()
answers = data["answer"].tolist()
@st.cache_resource
def load_model():
    model = SentenceTransformer(
        'sentence-transformers/all-MiniLM-L6-v2'
    )
    return model
model = load_model()
@st.cache_resource
def create_embeddings():
    embeddings = model.encode(questions)
    return embeddings
question_embeddings = create_embeddings()
def chatbot(user_input):
    user_embedding = model.encode([user_input])
    similarities = cosine_similarity(
        user_embedding,
        question_embeddings
    )
    best_match = similarities.argmax()
    confidence = similarities[0][best_match]
    if confidence < 0.40:
        return (
            "I'm sorry, I could not understand your question clearly. "
            "Please ask about Noonan syndrome symptoms, causes, diagnosis, or treatment."
        )
    return answers[best_match]
if "messages" not in st.session_state:
    st.session_state.messages = []
user_input = st.chat_input(
    "Ask a medical question..."
)
if user_input:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )
    response = chatbot(user_input)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div class="chat-user">
            👤 <b>You:</b><br><br>
            {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="chat-bot">
            🤖 <b>Medical AI:</b><br><br>
            {message["content"]}
            </div>
            """,
            unsafe_allow_html=True
        )
st.markdown("---")
st.caption(
    "Developed using Python, Streamlit, Sentence Transformers, and HuggingFace"
)
