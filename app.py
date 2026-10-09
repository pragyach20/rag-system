import os
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


# Load API key
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file.")


# Use the local course document when available; otherwise use public demo data.
BASE_DIR = Path(__file__).resolve().parent
private_document = BASE_DIR / "documents" / "company_knowledge.txt"
sample_document = BASE_DIR / "sample_data" / "company_knowledge.txt"
knowledge_base_path = (
    private_document if private_document.exists() else sample_document
)

if not knowledge_base_path.exists():
    raise FileNotFoundError(
        "No knowledge-base file found. Provide the local course document "
        "or keep sample_data/company_knowledge.txt in the project."
    )

loader = TextLoader(str(knowledge_base_path), encoding="utf-8")

documents = loader.load()

print(f"Loaded {len(documents)} document(s).")


# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} text chunks.")


# Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


# Connect to ChromaDB
vector_store = Chroma(
    collection_name="technova_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# Add documents only if the database is empty
if vector_store._collection.count() == 0:
    vector_store.add_documents(chunks)
    print("Documents added to ChromaDB.")
else:
    print("Connected to existing ChromaDB.")


# Create retriever
retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# Create Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


# RAG prompt
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using ONLY the information
provided in the context below.

If the answer is not available in the context,
say:

"I don't have enough information in the knowledge base
to answer that question."

Do not make up information.

Context:
{context}

Question:
{question}
"""
)


# Get user question
question = input("\nAsk a question about the company: ")


# Retrieve relevant information
retrieved_docs = retriever.invoke(question)

context = "\n\n".join(
    document.page_content
    for document in retrieved_docs
)


# Generate answer
messages = prompt.invoke(
    {
        "context": context,
        "question": question
    }
)

response = llm.invoke(messages)


# Clean response
if isinstance(response.content, list):
    answer = "".join(
        item.get("text", "")
        for item in response.content
        if isinstance(item, dict)
    )
else:
    answer = response.content


# Display answer
print("\n" + "=" * 60)
print("RAG ANSWER")
print("=" * 60)

print(answer)


# Display sources
print("\n" + "=" * 60)
print("RETRIEVED SOURCES")
print("=" * 60)

for i, document in enumerate(retrieved_docs, start=1):
    print(f"\nSource {i}:")
    print(document.page_content)