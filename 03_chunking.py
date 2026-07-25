"""
03_chunking.py
المرحلة الثالثة: تقسيم النصوص إلى Chunks (وحدات نصية) جاهزة للتمثيل المتجهي.

مستندات أسوان قصيرة أصلاً (جملة إلى ثلاث جمل)، فغالباً كل مستند يتحوّل إلى chunk واحد،
لكن الدالة chunk_text تدعم تقسيم أي نص أطول تلقائياً بحد أقصى لعدد الكلمات مع تداخل بسيط
(overlap) بين القطع لتفادي فقدان السياق عند حدود التقسيم.
"""
import importlib

preprocessing_module = importlib.import_module("02_preprocessing")


def chunk_text(text, max_words=60, overlap=15):
    """يقسّم نصاً طويلاً إلى قطع بحد أقصى max_words كلمة، مع تداخل overlap كلمة بين كل قطعة والتالية."""
    words = text.split()
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + max_words
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks(documents=None):
    """يبني قائمة الـ chunks النهائية من كل المستندات، مع الاحتفاظ بمعلومات المستند الأصلي."""
    documents = documents if documents is not None else preprocessing_module.preprocess_documents()
    chunks = []
    chunk_id = 0
    for doc in documents:
        pieces = chunk_text(doc["text"])
        for piece in pieces:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": doc["id"],
                    "category": doc["category"],
                    "text": piece,
                }
            )
            chunk_id += 1
    return chunks


if __name__ == "__main__":
    chunks = build_chunks()
    print(f"عدد الـ chunks الناتجة: {len(chunks)}")
    print("مثال على chunk:")
    print(chunks[0])
