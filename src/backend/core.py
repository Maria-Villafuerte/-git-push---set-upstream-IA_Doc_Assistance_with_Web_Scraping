import os
from pinecone import Pinecone
from typing import Any, Dict, List
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Pinecone as PineconeLangChain

load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])


def run_llm(query: str, chat_history: List[Dict[str, Any]] = []) -> Any:
    embeddings = OpenAIEmbeddings()

    # Get the Pinecone index
    index = pc.Index(os.environ['INDEX_NAME'])

    # Create the PineconeLangChain instance
    docsearch = PineconeLangChain(index, embeddings.embed_query, "text")

    chat = ChatOpenAI(verbose=True, temperature=0)

    # Create a local prompt template
    retrieval_qa_prompt = ChatPromptTemplate.from_template("""
    Answer the following input based on the given context:

    Context: {context}

    Question: {input}

    Answer: """)

    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_prompt)

    # Create a local prompt template for rephrasing
    rephrase_prompt = ChatPromptTemplate.from_template("""
    Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

    Chat History:
    {chat_history}

    Follow Up Input: {input}

    Standalone question:""")

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )

    qa = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=stuff_documents_chain
    )
    result = qa.invoke({"input": query, "chat_history": chat_history})
    new_result = {
        "query": result["input"],
        "result": result["answer"],
        "source": result["context"],
    }

    return new_result


if __name__ == '__main__':
    print(run_llm(query="what is Chain in LangChain?"))