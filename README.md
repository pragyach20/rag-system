# AI-Powered RAG System

A Retrieval-Augmented Generation (RAG) application built with Python, LangChain, Google Gemini, ChromaDB, and Gradio.

## Features

- Load and process text documents
- Split documents into smaller chunks
- Generate embeddings using Google Gemini
- Store and retrieve vectors using ChromaDB
- Perform semantic similarity search
- Generate answers grounded in retrieved context
- Provide an interactive Gradio web interface
- Keep API credentials in environment variables
- Use fictional sample data when the private local document is unavailable

## Tech Stack

- Python
- LangChain
- Google Gemini API
- ChromaDB
- Gradio
- python-dotenv

## Project Structure

- `app.py` - Loads documents and prepares the vector database
- `gradio_app.py` - Provides the interactive question-answering interface
- `sample_data/company_knowledge.txt` - Fictional sample knowledge base
- `requirements.txt` - Python dependencies
- `.env.example` - Example environment configuration

The local `documents/` directory, `.env`, and generated `chroma_db/` are excluded from GitHub.

## Setup

1. Install dependencies:

   `py -m pip install -r requirements.txt`

2. Create a `.env` file by copying `.env.example`.

3. Add your Gemini API key to `.env`:

   `GOOGLE_API_KEY=your_actual_google_api_key`

4. Prepare the vector database:

   `py app.py`

5. Start the web interface:

   `py gradio_app.py`

Open the local URL printed by Gradio in your browser.

## Knowledge Base

The application uses `documents/company_knowledge.txt` when that local file exists. Otherwise, it uses the fictional example at `sample_data/company_knowledge.txt`.

The included sample data is invented for demonstration purposes.

## Security

Never commit your real `.env` file, API keys, private documents, or generated vector database. Review additional files before publishing the project publicly.

## Author

Pragya Chaudhary
