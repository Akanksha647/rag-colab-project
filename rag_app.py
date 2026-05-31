from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import pipeline

# 1. Load documents
loader = TextLoader("data/sample.txt")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)

# 3. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 4. Store in FAISS vector DB
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 6. Use small local model
generator = pipeline(
    "text-generation",
    model="distilgpt2"
    max_new_tokens=200
)

def ask_question(question):
    relevant_docs = retriever.invoke(question)
    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
Answer the question using only the context below.

Context:
{context}

Question:
{question}

Answer:
"""

    result = generator(prompt)
    return result[0]["generated_text"].replace(prompt, "").strip()

if __name__ == "__main__":
    question = "What is RAG?"
    answer = ask_question(question)
    print(answer)