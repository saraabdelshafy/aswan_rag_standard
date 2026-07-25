"""
04_vector_representation.py
المرحلة الرابعة: تحويل نصوص الـ chunks إلى متجهات (Embeddings) باستخدام
نموذج Sentence-Transformer متعدد اللغات يدعم العربية.
"""
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def get_model():
    """يحمّل نموذج التمثيل المتجهي مرة واحدة فقط (Lazy Loading) ويعيد استخدامه بعد ذلك."""
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def embed_texts(texts):
    """يحوّل قائمة نصوص إلى قائمة متجهات (normalized) جاهزة للتخزين أو المقارنة."""
    model = get_model()
    vectors = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)
    return vectors.tolist()


if __name__ == "__main__":
    import importlib

    chunking_module = importlib.import_module("03_chunking")
    chunks = chunking_module.build_chunks()
    sample_texts = [c["text"] for c in chunks[:3]]
    vectors = embed_texts(sample_texts)
    print(f"عدد المتجهات: {len(vectors)} | طول كل متجه: {len(vectors[0])}")
