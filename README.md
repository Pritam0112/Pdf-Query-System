# Interactive PDF Query System with LangChain & Groq
This project is a Streamlit-based web application that allows users to upload PDF documents and ask questions about their content using a conversational AI interface. The system uses advanced natural language processing techniques to understand and respond to user queries based on the content of the uploaded PDFs.

## Features

- PDF document upload and processing
- Conversational interface for querying document content
- Uses FAISS for efficient vector storage and retrieval
- Powered by the Groq language model for natural language understanding and generation
- Session-based chat history

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Pritam0112/pdf-query-system.git
   cd pdf-query-system
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration

1. Create a `.streamlit/secrets.toml` file in the project root directory.
2. Add your API keys to the file:
   ```
   HF_TOKEN = "your_huggingface_token"
   GROQ_API_KEY = "your_groq_api_key"
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and go to the URL displayed in the terminal (usually `http://localhost:8501`).

3. Use the sidebar to upload PDF documents and process them.

4. Once processed, you can start asking questions about the content of the PDFs in the main chat interface.

   [Streamlit App](https://pdf-query-system-pritam-pohankar.streamlit.app/)

## Development

This project uses:
- Streamlit for the web interface
- LangChain for building the conversational AI pipeline
- FAISS for vector storage
- Groq's language model for natural language processing
