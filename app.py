import streamlit as st
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore

# ------------------- Extract Text -------------------
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")  # Use fitz to open the PDF
    text = "\n".join(page.get_text() for page in doc)  # Extract text from all pages
    return text

# ------------------- Vector Store -------------------
@st.cache_resource
def create_vectorstore(text):
    # Split the text into chunks
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])
    
    # Embed the documents
    embed_model = OllamaEmbeddings(model="nomic-embed-text")
    embeddings = embed_model.embed_documents([doc.page_content for doc in docs])
    
    # Convert embeddings to numpy array (FAISS requires float32 type)
    embeddings = np.array(embeddings).astype('float32')
    
    # Create FAISS index
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance index
    index.add(embeddings)
    
    # Create an InMemoryDocstore instead of a dict
    docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})

    # Map index to document ID
    index_to_docstore_id = {i: i for i in range(len(docs))}

    # Create FAISS vector store with docstore and index_to_docstore_id
    vectorstore = FAISS(index=index, embedding_function=embed_model, docstore=docstore, index_to_docstore_id=index_to_docstore_id)

    # Return the FAISS vectorstore
    return vectorstore

# ------------------- QA Chain -------------------
def make_qa_chain(text):
    vectorstore = create_vectorstore(text)
    retriever = vectorstore.as_retriever()  # Get the retriever from FAISS vectorstore
    llm = Ollama(model="llama3")
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ------------------- Streamlit App -------------------
st.set_page_config(page_title="A2A Career Coach", layout="wide")
st.title("👔 A2A Career Coach")
st.caption("Upload your resume and get AI-powered insights.")

uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type="pdf")

if uploaded_file:
    text = extract_text_from_pdf(uploaded_file)
    st.success("✅ Resume Uploaded Successfully")
    qa_chain = make_qa_chain(text)

    question = st.chat_input("Ask a career question based on your resume...")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if question:
        answer = qa_chain.run(question)
        st.session_state.chat_history.append((question, answer))

    for q, a in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            st.markdown(a)

    # 📈 Skill Frequency Visualization
    st.divider()
    st.subheader("📊 Skill Frequency Insight")
    with st.expander("Show Skill Insights"):
        words = [w for w in text.split() if len(w) > 4]
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1
        top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]
        labels, values = zip(*top)
        fig, ax = plt.subplots()
        sns.barplot(x=list(values), y=list(labels), ax=ax)
        st.pyplot(fig)

    # ✍️ Blog Generation (Enhanced)
    st.divider()
    if st.button("✍️ Generate Career Advice Blog"):
        # Generate the blog prompt based on resume content
        blog_prompt = f"""
        Write a detailed career advice blog based on the following resume. Focus on:
        - Suggesting suitable job roles.
        - Highlighting areas for skill improvements.
        - Recommending industries to target.
        - Offering tips for career growth and next steps.
        Resume: {text}
        """
        try:
            blog = qa_chain.run(blog_prompt)
            st.text_area("AI-Generated Career Advice Blog:", blog, height=300)
        except Exception as e:
            st.error(f"Error generating career blog: {str(e)}")

else:
    st.info("⬅️ Upload a resume PDF to get started.")
    st.markdown(
        """
        - **Ask questions** about your resume.
        - **Get insights** on skills and experiences.
        - **Generate a blog** for career advice.
        """
    )
    st.markdown(
        """Made with ❤️ by [Krishthick](https://github.com/Krishthick)"""
    )
    st.markdown(
        """
        **Disclaimer:** This is a demo application. Please do not share sensitive information.
        """
    )
    st.markdown(
        """
        **Note:** This application uses the Ollama API for LLM and embedding services. Ensure you have the necessary API keys and permissions.
        """
    )
    st.markdown(
        """
        **License:** This application is licensed under the MIT License. See [LICENSE](LICENSE) for details.
        """
    )
