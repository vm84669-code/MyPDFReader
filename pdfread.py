
pip install -U langchain-community

import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import PyPDF2
import tempfile
import os

# Set your Gemini API key
os.environ['GOOGLE_API_KEY'] = st.secrets['GOOGLE_API_KEY']

st.title("PDF Query Chatbot with Gemini")

# Step 1: PDF Upload and Text Extraction
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
pdf_text = ""

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(uploaded_file.read())
        temp_pdf_path = temp_pdf.name

    with open(temp_pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            pdf_text += page.extract_text()

    st.success("PDF loaded successfully!")

    # Step 2: Chunk Text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=64
    )
    docs = text_splitter.split_text(pdf_text)
    st.write(f"Document chunked into {len(docs)} parts.")

    # Step 3: Embedding and VectorDB
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-1.5-flash-latest")
    db = FAISS.from_texts(docs, embeddings)

    # Step 4: Chatbot User Query
    st.header("Ask a Question from your PDF")

    user_query = st.text_input("Enter your question:")

    if st.button("Get Answer") and user_query:
        # Search relevant chunks
        relevant_docs = db.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Prompt preparation
        prompt_template = """
        Answer the following question based only on the context from the PDF file.
        Context:
        {context}

        Question:
        {question}
        """
        template = PromptTemplate(
            input_variables=["context", "question"],
            template=prompt_template
        )

        # Initialize Gemini
        chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        chain = LLMChain(llm=chat_model, prompt=template)

        answer = chain.invoke({"context": context, "question": user_query})
        st.write(answer.get('content', 'No answer returned.'))
