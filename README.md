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

## تشغيل محلي

```bash
pip install -r requirements.txt

# اختياري لتجربة التوليد محلياً (لا يُرفع أبداً على GitHub):
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# ثم عدّلي OPENROUTER_API_KEY داخل الملف بمفتاحك الحقيقي

streamlit run streamlit_app.py
```

## الحصول على مفتاح OpenRouter (مجاني)

1. سجّلي في **openrouter.ai** (تسجيل مجاني).
2. من **Keys** أنشئي مفتاحاً جديداً.
3. النموذج الافتراضي هنا `meta-llama/llama-3.1-8b-instruct:free` — نموذج **مجاني بالكامل** (لاحقة
   `:free`)، يعني تقدري تجربي وتسلّمي المشروع بدون أي تكلفة أو بطاقة بنكية.

## قاعدة API — الأمان

- لا يوجد أي مفتاح API مكتوب داخل أي ملف بايثون (يُقرأ فقط من متغيرات البيئة أو Secrets).
- ملف `.streamlit/secrets.toml` الحقيقي مستبعد من Git عبر `.gitignore` ولا يُرفع أبداً.
- الموجود في المستودع هو `secrets.toml.example` فقط (قالب فارغ بدون مفتاح حقيقي).

## إعداد Streamlit Secrets عند النشر

1. بعد نشر التطبيق على Streamlit Cloud، من صفحته اضغطي **Manage app**.
2. افتحي **Secrets**.
3. الصقي بالضبط:
```toml
OPENROUTER_API_KEY = "مفتاحك_الحقيقي_هنا"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"
```
4. احفظي — التطبيق هيقرأها تلقائياً بدون أي تعديل إضافي في الكود.

## رفع المشروع على GitHub

```bash
git init
git add .
git commit -m "Aswan RAG project - final structure"
git branch -M main
git remote add origin <رابط المستودع بتاعك>
git push -u origin main
```

تأكدي إن `chroma_db/` و `.streamlit/secrets.toml` **متضافين مش موجودين** في الرفع (`.gitignore`
بيستبعدهم تلقائياً).

## نشر على Streamlit Community Cloud

1. share.streamlit.io → Sign in with GitHub → New app
2. Repository: مستودعك، Branch: `main`
3. **Main file path:** `streamlit_app.py`
4. Deploy، ثم أضيفي الـ Secrets كما بالأعلى.

## قائمة التحقق النهائية

- [x] كل الملفات المطلوبة موجودة (`01_` إلى `07_` + `streamlit_app.py` + `requirements.txt`)
- [x] لا يوجد مفتاح API حقيقي داخل أي ملف بايثون
- [x] لا يوجد ملف `.env` حقيقي مرفوع
- [x] Secrets بصيغة TOML صحيحة
- [x] الإجابة مبنية على السياق المسترجع فعلياً (لا اختلاق)
- [x] الإجابة تذكر مصادرها دائماً (قسم "المصادر المستخدمة" يُلحق تلقائياً بكل إجابة)
