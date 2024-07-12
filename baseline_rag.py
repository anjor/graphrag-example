import os
from openai import OpenAI
import time
from pinecone import Index, Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

# Set your API keys
client = OpenAI()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    end = chunk_size

    while start < len(text):
        last_period = text.rfind(".", start, end)

        if last_period != -1:
            chunk = text[start : last_period + 1]
        else:
            chunk = text[start:end]

        chunks.append(chunk)

        start = max(start + chunk_size - overlap, last_period + 1)
        end = start + chunk_size

    return chunks


def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=text, model=model).data[0].embedding


def create_rag_application(file_path, chunk_size=1000, chunk_overlap=200):
    # 1. Load and chunk the text
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read()

    chunks = chunk_text(text, chunk_size, chunk_overlap)

    # 2. Create or connect to Pinecone index
    index_name = "pdf-embeddings"
    if index_name not in pc.list_indexes():
        pc.create_index(
            index_name,
            dimension=1536,
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )  # OpenAI embeddings have 1536 dimensions
    index = pc.Index(index_name)

    # 3. Create embeddings and upsert to Pinecone
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        index.upsert(vectors=[(str(i), embedding, {"text": chunk})])
        time.sleep(0.1)  # To avoid hitting rate limits

    return index, chunks


def query_rag(index: Index, query, k=3):
    # 1. Embed the query
    query_embedding = get_embedding(query)

    # 2. Query Pinecone
    results = index.query(vector=query_embedding, top_k=k, include_metadata=True)

    # 3. Retrieve relevant chunks
    relevant_chunks = [result.metadata["text"] for result in results.matches]

    # 4. Prepare context for OpenAI
    context = "\n\n".join(relevant_chunks)

    # 5. Query OpenAI
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use the following context to answer the user's question.",
            },
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"},
        ],
    )

    return response.choices[0].message.content, relevant_chunks


if __name__ == "__main__":
    # file_path = "./ragtest/input/extracted_text.txt"
    # index, chunks = create_rag_application(file_path)

    # print(index)

    user_query = "What are the main themes of this article?"
    index = pc.Index("pdf-embeddings")

    answer, sources = query_rag(index, user_query)
    print(f"\nAnswer: {answer}\n")
    print("Sources:")
    for i, chunk in enumerate(sources, 1):
        print(f"{i}. {chunk[:100]}...")
    print()
