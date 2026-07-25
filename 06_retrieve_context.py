"""
06_retrieve_context.py
المرحلة السادسة: استرجاع أقرب الـ chunks لسؤال المستخدم من مخزن Chroma
عن طريق تحويل السؤال لمتجه ومقارنته بمتجهات الـ chunks المخزّنة.
"""
import importlib

store_module = importlib.import_module("05_create_chroma_store")
vector_module = importlib.import_module("04_vector_representation")

_collection_cache = {}


def _get_collection(persist=True):
    """يبني مخزن Chroma مرة واحدة فقط لكل نوع (persist/in-memory) ويعيد استخدامه بعد ذلك."""
    key = "persist" if persist else "memory"
    if key not in _collection_cache:
        _, collection = store_module.build_store(persist=persist)
        _collection_cache[key] = collection
    return _collection_cache[key]


def retrieve_context(query, k=3, persist=True):
    """يرجع أفضل k قطعة نصية (chunks) الأقرب دلالياً لسؤال المستخدم."""
    collection = _get_collection(persist=persist)
    query_embedding = vector_module.embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    context = []
    for i in range(len(results["ids"][0])):
        context.append(
            {
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "category": results["metadatas"][0][i].get("category"),
                "document_id": results["metadatas"][0][i].get("document_id"),
                "distance": results["distances"][0][i],
            }
        )
    return context


if __name__ == "__main__":
    query = "عادات وتقاليد أهل النوبة في الاحتفالات"
    context = retrieve_context(query, k=3)
    print(f"السؤال: {query}\n")
    for c in context:
        print(f"[مصدر {c['document_id']}] ({c['category']}) distance={c['distance']:.3f}")
        print(c["text"], "\n")
