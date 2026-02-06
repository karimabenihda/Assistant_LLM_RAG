from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.retrieval_qa.base import RetrievalQA

def build_qa_chain(persist_dir, prompt_path):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", # Vérifie la version (2.5 n'existe pas encore en 2024/25)
        temperature=0
    )

    with open(prompt_path, "r", encoding="utf-8") as f:
        prompt_text = f.read()

    prompt = PromptTemplate(
        template=prompt_text,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 6}
        ),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )