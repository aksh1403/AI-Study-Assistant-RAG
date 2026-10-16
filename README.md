# Ask Your PDF

A Python application that lets users upload a PDF and ask questions about its contents using natural language.

The application uses Google Gemini, LangChain, Streamlit, and FAISS to implement a basic Retrieval-Augmented Generation (RAG) pipeline.

## Features

* Upload PDF documents.
* Extract text from PDF pages.
* Split text into smaller chunks.
* Generate embeddings using Google Gemini.
* Store and retrieve relevant chunks using FAISS.
* Generate answers using a Gemini language model.
* Inspect the retrieved text used to answer questions.
* Cache processed documents to avoid rebuilding the vector store for every question.

## How It Works

1. The application extracts text from the uploaded PDF.
2. The text is split into smaller chunks.
3. Gemini generates embeddings for the chunks.
4. FAISS stores the embeddings and retrieves relevant chunks for each question.
5. The retrieved context and question are passed to Gemini.
6. The application displays the generated answer.

This is a basic RAG implementation. Answer accuracy depends on document extraction, retrieval quality, and the language model.

## Requirements

* Python 3.10 or a compatible Python version supported by the installed dependencies.
* A Google Gemini API key.
* The packages listed in `requirements.txt`.

## Installation

Clone the repository and navigate to the project directory.

Install the dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```text
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

Replace the placeholder with your own API key.

Never commit `.env` to GitHub.

## Usage

Run the application:

```bash
streamlit run app.py
```

Upload a text-based PDF, enter a question about its contents, and review the generated answer.

Scanned PDFs may require OCR before their text can be extracted.

## Technologies Used

* Python
* Streamlit
* LangChain
* Google Gemini
* Google Gemini Embeddings
* FAISS
* pypdf

## Limitations

* The application extracts text rather than performing OCR.
* Answers depend on the relevance of retrieved chunks.
* Conversation history is not currently supported.
* Retrieved context may not contain all the information needed to answer a question.
* The application has not been configured with a formal evaluation system.

## Security

Keep API keys private. Store secrets in environment variables and never commit them to version control.
