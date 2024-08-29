import os
import sqlite3
import logging
import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check SQLite version
try:
    conn = sqlite3.connect(':memory:')
    logging.info(f"SQLite version: {sqlite3.sqlite_version}")
except Exception as e:
    logging.error(f"Error: {e}")

# Load environment variables
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]
api_key = st.secrets["GROQ_API_KEY"]

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_pdf_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = "./temp.pdf"
        with open(temp_pdf, "wb") as file:
            file.write(uploaded_file.getvalue())
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)
    return documents

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    return vectorstore

def create_retriever(vectorstore):
    retriever = vectorstore.as_retriever()
    return retriever

def setup_conversational_chain(llm, retriever):
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise.\n\n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

def main():
    st.title("Conversational RAG With PDF uploads and chat history")
    st.write("Upload PDFs and chat with their content")

    llm = ChatGroq(groq_api_key=api_key, model_name="Gemma2-9b-It")

    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Process Button", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing"):
                    documents = get_pdf_documents(pdf_docs)
                    vectorstore = create_vector_store(documents)
                    retriever = create_retriever(vectorstore)
                    st.session_state.conversation = setup_conversational_chain(llm, retriever)
                    st.success("Processing Complete")
            else:
                st.warning("Please upload at least one PDF file.")

    session_id = st.text_input("Session ID", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    if "conversation" in st.session_state:
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What's on your mind?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Process the prompt through the conversational chain
            conversational_rag_chain = RunnableWithMessageHistory(
                st.session_state.conversation, get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
            response = conversational_rag_chain.invoke(
                {"input": prompt},
                config={"configurable": {"session_id": session_id}}
            )

            # Display assistant response in chat message container
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            with st.chat_message("assistant"):
                st.markdown(response['answer'])

    else:
        st.warning("Please upload and process a PDF file in the sidebar.")

if __name__ == "__main__":
    main()
