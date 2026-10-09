import os
import gradio as gr
from dotenv import load_dotenv

from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY is not set in the .env file.")


embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=api_key
)


vector_store = Chroma(
    collection_name="technova_knowledge",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0
)


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


def answer_question(question):
    if not question.strip():
        return "Please enter a question.", ""

    try:
        retrieved_docs = retriever.invoke(question)

        context = "\n\n".join(
            document.page_content
            for document in retrieved_docs
        )

        messages = prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = llm.invoke(messages)

        if isinstance(response.content, list):
            answer = "".join(
                item.get("text", "")
                for item in response.content
                if isinstance(item, dict)
            )
        else:
            answer = response.content

        sources = "\n\n".join(
            f"Source {i}:\n{document.page_content}"
            for i, document in enumerate(
                retrieved_docs,
                start=1
            )
        )

        return answer, sources

    except Exception as e:
        return f"Error: {str(e)}", ""


def clear_fields():
    return "", "", ""


with gr.Blocks(title="TechNova RAG Assistant") as demo:

    gr.Markdown(
        """
        # 🤖 TechNova RAG Assistant

        Ask questions about the TechNova Solutions knowledge base.

        **Powered by:** LangChain · Google Gemini · ChromaDB · RAG
        """
    )

    question = gr.Textbox(
        label="Ask a Question",
        placeholder="Example: What are the company's working hours?",
        lines=2
    )

    with gr.Row():
        ask_button = gr.Button(
            "Ask Question",
            variant="primary"
        )

        clear_button = gr.Button("Clear")

    gr.Examples(
        examples=[
            ["What are the company's working hours?"],
            ["How many paid leaves do employees receive?"],
            ["What is the work-from-home policy?"],
            ["Who should employees contact for salary questions?"],
            ["What technologies does the company use?"],
        ],
        inputs=question
    )

    answer = gr.Textbox(
        label="AI Answer",
        lines=5
    )

    sources = gr.Textbox(
        label="Retrieved Sources",
        lines=12
    )

    ask_button.click(
        fn=answer_question,
        inputs=question,
        outputs=[answer, sources]
    )

    question.submit(
        fn=answer_question,
        inputs=question,
        outputs=[answer, sources]
    )

    clear_button.click(
        fn=clear_fields,
        outputs=[question, answer, sources]
    )


if __name__ == "__main__":
    demo.launch()