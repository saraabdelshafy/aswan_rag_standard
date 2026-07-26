# 🏛️ أسوان RAG — نسخة متوافقة مع متطلبات المشروع

نظام RAG (Retrieval-Augmented Generation) عن ثقافة ومعالم أسوان، مبني بالكامل بملفات بايثون
منفصلة (بدون Notebooks) حسب الخط المطلوب:

```
documents -> preprocessing -> chunking -> vector representation
-> vector store -> context retrieval -> prompting -> Streamlit UI
```

## هيكل الملفات

| الملف | المرحلة |
|---|---|
| `01_documents.py` | قاعدة المعرفة الخام (51 مستنداً أصلياً عن أسوان) |
| `02_preprocessing.py` | تنظيف وتوحيد النصوص العربية |
| `03_chunking.py` | تقسيم النصوص إلى Chunks |
| `04_vector_representation.py` | تحويل النصوص لمتجهات (Sentence-Transformers) |
| `05_create_chroma_store.py` | بناء مخزن متجهات بـ **ChromaDB** |
| `06_retrieve_context.py` | استرجاع أقرب Chunks لسؤال المستخدم |
| `07_prompting.py` | بناء الـ Prompt + التوليد عبر **OpenRouter API** + ذكر المصادر |
| `streamlit_app.py` | الواجهة النهائية (شات تفاعلي) |
| `requirements.txt` | المكتبات المطلوبة |

كل ملف مرقّم يعمل أيضاً بشكل مستقل لعرض مخرجات مرحلته:
```bash
python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py
python 05_create_chroma_store.py
python 06_retrieve_context.py
python 07_prompting.py
```
