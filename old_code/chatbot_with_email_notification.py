#to start di application: streamlit run chatbot.py
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
import os 
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
import atexit
import shutil
from datetime import datetime
import smtplib
import ssl
from email.message import EmailMessage

# Set OpenAI key (prefer Streamlit secrets for security)
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
else:
    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")

# --- Email notification (optional) ---
def send_usage_email(user_question: str) -> None:
    """Send a short email when someone submits a question.

    Configure via Streamlit secrets:
      SMTP_HOST, SMTP_PORT (optional, default 465),
      SMTP_USERNAME, SMTP_PASSWORD,
      NOTIFY_TO, NOTIFY_FROM (or use SMTP_USERNAME as from).
    """
    try:
        secrets = st.secrets
    except Exception:
        secrets = {}

    smtp_host = secrets.get("SMTP_HOST") or os.getenv("SMTP_HOST")
    smtp_port = int(secrets.get("SMTP_PORT") or os.getenv("SMTP_PORT") or 465)
    smtp_user = secrets.get("SMTP_USERNAME") or os.getenv("SMTP_USERNAME")
    smtp_pass = secrets.get("SMTP_PASSWORD") or os.getenv("SMTP_PASSWORD")

    notify_to = secrets.get("NOTIFY_TO") or os.getenv("NOTIFY_TO")
    notify_from = secrets.get("NOTIFY_FROM") or os.getenv("NOTIFY_FROM") or smtp_user

    # If not configured, do nothing
    if not all([smtp_host, smtp_user, smtp_pass, notify_to, notify_from]):
        return

    msg = EmailMessage()
    msg["Subject"] = "App usage notification, new question submitted"
    msg["From"] = notify_from
    msg["To"] = notify_to
    msg.set_content(
        "A user submitted a question in the Streamlit app.\n\n"
        f"Time (UTC): {datetime.utcnow().isoformat()}\n\n"
        f"Question:\n{user_question.strip()}\n"
    )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        server.login(smtp_user, smtp_pass)
        server.send_message(msg)

# --- Function to save discussion log on exit ---
def save_discussion_on_exit():
    """
    Copies the last session's discussion.txt to a dated file with a timestamp.
    This function is triggered when the Streamlit server process is stopped (e.g., via Ctrl+C).
    It saves the log from the last active session before shutdown.
    """
    source_path = "discussion.txt"
    if os.path.exists(source_path) and os.path.getsize(source_path) > 0:
        # Create a unique filename with a timestamp to prevent overwriting
        timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        destination_path = f"discussion_log_{timestamp_str}.txt"
        try:
            shutil.copy(source_path, destination_path)
            # This print is for the server console, not the user UI
            print(f"INFO: Discussion log from session saved to {destination_path}")
        except Exception as e:
            print(f"ERROR: Could not save discussion log. Reason: {e}")

# Register the function to be called when the script exits
atexit.register(save_discussion_on_exit)

# Page config
st.set_page_config(
    page_title="Berardino Barile – AI Experience Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Sidebar with links and profile
with st.sidebar:
    st.image("Berardino.jpeg", width=200, output_format="auto")
    st.markdown("### 👋 Hi, I'm Berardino")
    st.markdown("Postdoctoral Researcher at McGill & Mila AI Institute")
    st.markdown("""
    🔬 Specializing in:
    - Causal ML
    - Generative AI
    - Statistics
    - Machine Learning
    - Data Science
    - Optimization
    - Probability Theory
    - Reinforcement Learning
    - Deep Learning
    - Graph Neural Networks
    - Bayesian Inference
    - Large Language Models
    - Computer Vision
    - Time Series Analysis
    - Causal Inference
    - Statistical Learning
    """)
    st.markdown("🌐 [PVG Page](https://www.cim.mcgill.ca/~bbera/)")
    st.markdown("💼 [LinkedIn](https://www.linkedin.com/in/berardino-barile/)")
    st.markdown("🐙 [GitHub](https://github.com/BerardinoB?tab=repositories)")
    st.markdown("📖 [Google Scholar](https://scholar.google.com/citations?user=odmpMGcAAAAJ&hl=en)")
    st.markdown("🖥️ [Website](https://berardinob.github.io/BNormal/)")
    st.markdown("---")

# Title and instructions
st.markdown("<h1 style='text-align: center;'>💼 Ask Me About My Work Experience</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>This chatbot is trained on my personal experience. Ask me anything!</p>", unsafe_allow_html=True)

@st.cache_resource(show_spinner="Loading knowledge base...")
def build_vectorstore():
    # Load the same embedding model used when saving the index
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorstore

def make_qa_chain(vectorstore: FAISS):
    """Create the QA chain with a custom prompt."""
    # Retrieve more documents (k=5) to give the LLM more context to synthesize from.
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})

    prompt_template = """
You are a professional assistant describing the work experience of Berardino Barile.
You must always speak in the third person. Never use first-person pronouns like "I" or "my".
Always refer to the subject as "Berardino" or "Berardino Barile". Avoid using generic terms like "the individual" or "he".

Use the information provided in the 'Context' below to answer the question. Synthesize details from all relevant parts of the context to provide a comprehensive and detailed answer.
If the information to answer the question is not in the context, you must respond with the following exact phrase (do not replece I with Berardino in this case): "I cannot provide you with this information. Please write an email directly to Berardino at: berardino.barile@gmail.com".
Do not add any other words or explanations if the information is not found.

Context: {context}
Question: {question}
Answer:
"""
    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": custom_prompt}
    )

def log_interaction(question: str, answer: str):
    """Append the Q&A to the discussion log and today’s daily log file."""
    entry = f"Question: {question}\nAnswer: {answer}\n{'-'*40}\n"
    # Append to the main discussion file
    with open("discussion.txt", "a", encoding="utf-8") as f:
        f.write(entry)
    
    # Determine the daily log file name using the current date (day-month-year)
    date_str = datetime.now().strftime("%d-%m-%Y")
    daily_log_file = f"discussion_log_{date_str}.txt"
    
    # Append (or create) the daily log file with the new entry
    with open(daily_log_file, "a", encoding="utf-8") as f:
        f.write(entry)

def add_qa_to_vectorstore(vectorstore: FAISS, question: str, answer: str):
    """Add a new Q&A pair to the in-memory vector store."""
    content = f"Question: {question}\nAnswer: {answer}"
    doc = Document(page_content=content, metadata={"source": "session_qa"})
    vectorstore.add_documents([doc])

# Initialize vector store and QA chain in session state
if "vectorstore" not in st.session_state:
    # For a new session, clear the discussion log to only track current Q&A
    with open("discussion.txt", "w", encoding="utf-8") as f:
        f.write("")

    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = build_vectorstore()
        st.session_state.qa_chain = make_qa_chain(st.session_state.vectorstore)

# Callback to clear the text input
def clear_text():
    st.session_state.query_input = ""

# Chat input layout
query = st.text_input("💬 Ask a question about my experience:", key="query_input")

col1, col2, _ = st.columns([1, 1, 8])
with col1:
    ask_btn = st.button("Ask", type="primary")
with col2:
    st.button("Clear", on_click=clear_text)

if ask_btn and query.strip():
    # Notify you when someone uses the app (if SMTP is configured)
    send_usage_email(query)

    # Initialize counter if not available
    if "qa_count" not in st.session_state:
        st.session_state.qa_count = 0
    st.session_state.qa_count += 1

    if st.session_state.qa_count > 5:
        # After 5 questions, center the meme image and show a thank you message
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("assets/meme_me.jpg", width=300, caption="")
        st.markdown(
            "<p style='text-align: center; font-size: 16px;'>Thank you for your interest in my CV. Since you've asked over 5 questions, "
            "to limit the number of requests, I invite you to contact me directly.</p>",
            unsafe_allow_html=True
        )
    else:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain.run(query)
        st.markdown("### 🧠 Response")
        st.markdown(
            f"<div style='background-color:#f0f2f6;padding:16px;border-radius:12px;'>{response}</div>",
            unsafe_allow_html=True,
        )
        # Log the interaction and add it to the context for this session
        log_interaction(query, response)
        add_qa_to_vectorstore(st.session_state.vectorstore, query, response)

st.markdown("**Suggested questions:**")

suggested_questions = [
    "What is your current research focus?",
    "What experience do you have with vision language models?",
    "What projects have you worked on in medical AI?",
    "What are your strongest machine learning skills?",
    "What experience do you have with causal machine learning?",
]

cols = st.columns(3)
for i, q in enumerate(suggested_questions):
    with cols[i % 3]:
        if st.button(q, key=f"suggestion_{i}"):
            st.session_state.query_input = q


# Optional footer
st.markdown("---")
st.markdown("<small style='text-align:center;display:block;'>© 2025 Berardino Barile</small>", unsafe_allow_html=True)
