"""
05_create_chroma_store.py
المرحلة الخامسة: بناء مخزن متجهات (Vector Store) باستخدام ChromaDB،
وتخزين الـ chunks مع متجهاتها ومعلوماتها الوصفية (metadata) فيه.
"""
import importlib

import chromadb

chunking_module = importlib.import_module("03_chunking")
vector_module = importlib.import_module("04_vector_representation")

COLLECTION_NAME = "aswan_chunks"
PERSIST_DIR = "./chroma_db"


def build_store(persist=True, persist_dir=PERSIST_DIR):
    """
    يبني (أو يعيد استخدام) مجموعة Chroma ويملأها بالـ chunks ومتجهاتها.

    persist=True  -> يخزّن المجموعة على القرص في persist_dir (مناسب للتشغيل المحلي/الأسكريبت المستقل).
    persist=False -> عميل في الذاكرة فقط، يُعاد بناؤه في كل مرة (مناسب لتطبيق Streamlit المنشور).
    """
    client = chromadb.PersistentClient(path=persist_dir) if persist else chromadb.Client()
    collection = client.get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    if collection.count() == 0:
        chunks = chunking_module.build_chunks()
        texts = [c["text"] for c in chunks]
        embeddings = vector_module.embed_texts(texts)
        collection.add(
            ids=[str(c["chunk_id"]) for c in chunks],
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {"document_id": c["document_id"], "category": c["category"]} for c in chunks
            ],
        )

    return client, collection


if __name__ == "__main__":
    client, collection = build_store()
    print(f"عدد العناصر المخزّنة في مخزن Chroma: {collection.count()}")
    print(f"مسار التخزين: {PERSIST_DIR}")
