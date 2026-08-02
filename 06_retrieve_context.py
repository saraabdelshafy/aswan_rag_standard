"""
06_retrieve_context.py
المرحلة السادسة: استرجاع أقرب الـ chunks لسؤال المستخدم من مخزن Chroma
عن طريق تحويل السؤال لمتجه ومقارنته بمتجهات الـ chunks المخزّنة.

يتضمن هذا الملف مرحلة Context Filtering:
- فلترة النتائج بناءً على عتبة المسافة (distance threshold) لاستبعاد النتائج البعيدة دلالياً.
- إمكانية الفلترة بالفئة (category) عبر metadata.
- إزالة التكرارات على مستوى المستند الأصلي (اختياري).
"""
import importlib

store_module = importlib.import_module("05_create_chroma_store")
vector_module = importlib.import_module("04_vector_representation")

_collection_cache = {}

# عتبة المسافة الافتراضية: أي chunk مسافته أكبر من كده يعتبر بعيد دلالياً ويُستبعد.
# (المسافة cosine distance، فكل ما تكون أصغر يبقى الـ chunk أقرب للسؤال)
DEFAULT_MAX_DISTANCE = 0.8


def _get_collection(persist=True):
    """يبني مخزن Chroma مرة واحدة فقط لكل نوع (persist/in-memory) ويعيد استخدامه بعد ذلك."""
    key = "persist" if persist else "memory"
    if key not in _collection_cache:
        _, collection = store_module.build_store(persist=persist)
        _collection_cache[key] = collection
    return _collection_cache[key]


def filter_context(
    raw_results,
    max_distance=DEFAULT_MAX_DISTANCE,
    allowed_categories=None,
    deduplicate_by_document=False,
):
    """
    مرحلة Context Filtering: تصفية نتائج الاسترجاع الخام قبل تمريرها للنموذج.

    - max_distance: أقصى مسافة مسموح بها (يستبعد النتائج البعيدة دلالياً).
    - allowed_categories: قائمة الفئات المسموح بها فقط (None = كل الفئات).
    - deduplicate_by_document: لو True، يحتفظ بأفضل chunk فقط لكل document_id.
    """
    filtered = []
    seen_documents = set()

    for i in range(len(raw_results["ids"][0])):
        distance = raw_results["distances"][0][i]
        category = raw_results["metadatas"][0][i].get("category")
        document_id = raw_results["metadatas"][0][i].get("document_id")

        # فلتر 1: عتبة المسافة
        if distance > max_distance:
            continue

        # فلتر 2: الفئات المسموح بها
        if allowed_categories is not None and category not in allowed_categories:
            continue

        # فلتر 3: إزالة التكرار على مستوى المستند
        if deduplicate_by_document:
            if document_id in seen_documents:
                continue
            seen_documents.add(document_id)

        filtered.append(
            {
                "chunk_id": raw_results["ids"][0][i],
                "text": raw_results["documents"][0][i],
                "category": category,
                "document_id": document_id,
                "distance": distance,
            }
        )

    return filtered


def retrieve_context(query, k=3, persist=True, max_distance=0.8):
    """يرجع أفضل k قطعة نصية الأقرب دلالياً للسؤال، بشرط ألا تتجاوز المسافة القصوى المحددة."""
    collection = _get_collection(persist=persist)
    query_embedding = vector_module.embed_texts([query])[0]
    results = collection.query(query_embeddings=[query_embedding], n_results=k)

    context = []
    for i in range(len(results["ids"][0])):
        distance = results["distances"][0][i]
        
        # فلترة السياق: لا نضيف النص إلا لو كان قريب من السؤال
        if distance <= max_distance:
            context.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "category": results["metadatas"][0][i].get("category"),
                    "document_id": results["metadatas"][0][i].get("document_id"),
                    "distance": distance,
                }
            )
    return context


if __name__ == "__main__":
    query = "عادات وتقاليد أهل النوبة في الاحتفالات"
    print(f"السؤال: {query}\n")

    print("=" * 60)
    print("بدون فلترة (raw):")
    print("=" * 60)
    context_raw = retrieve_context(query, k=3, max_distance=999)
    for c in context_raw:
        print(f"[مصدر {c['document_id']}] ({c['category']}) distance={c['distance']:.3f}")
        print(c["text"], "\n")

    print("=" * 60)
    print(f"مع Context Filter (max_distance={DEFAULT_MAX_DISTANCE}, dedup=True):")
    print("=" * 60)
    context_filtered = retrieve_context(
        query, k=3, max_distance=DEFAULT_MAX_DISTANCE, deduplicate_by_document=True
    )
    for c in context_filtered:
        print(f"[مصدر {c['document_id']}] ({c['category']}) distance={c['distance']:.3f}")
        print(c["text"], "\n")
