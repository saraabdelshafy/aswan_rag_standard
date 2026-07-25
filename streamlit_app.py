"""
streamlit_app.py
واجهة Streamlit النهائية لمشروع أسوان RAG.

تجمع كل مراحل الخط: المستندات (01) -> التنظيف (02) -> التقطيع (03)
-> التمثيل المتجهي (04) -> مخزن Chroma (05) -> الاسترجاع (06) -> الـ Prompting (07)
-> عرض الإجابة مع مصادرها في واجهة شات تفاعلية.
"""
import importlib

import streamlit as st

# استيراد كل مراحل الـ pipeline (بالاعتماد على importlib لأن أسماء الملفات تبدأ بأرقام)
documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")
vector_module = importlib.import_module("04_vector_representation")
store_module = importlib.import_module("05_create_chroma_store")
retrieve_module = importlib.import_module("06_retrieve_context")
prompting = importlib.import_module("07_prompting")

# قراءة مفاتيح الإعداد من Streamlit Secrets عند النشر (بدون كتابة أي مفتاح حقيقي في الكود)
try:
    if not prompting.OPENROUTER_API_KEY:
        prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    prompting.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", prompting.OPENROUTER_MODEL)
except Exception:
    pass

st.set_page_config(page_title="أسوان RAG", page_icon="🏛️", layout="centered")
st.markdown(
    "<style>body, .stApp { direction: rtl; text-align: right; }</style>",
    unsafe_allow_html=True,
)

st.title("🏛️ أسوان RAG")
st.caption("نظام استرجاع وتوليد معزز بالمعرفة (RAG) عن ثقافة ومعالم أسوان")

with st.expander("ℹ️ عن المشروع"):
    st.write(
        f"قاعدة المعرفة تحتوي على {len(documents_module.get_documents())} مستنداً أصلياً عن أسوان "
        "(المعابد، الحدائق، النزهات النيلية، عادات وتقاليد النوبة). كل إجابة تُبنى فقط من "
        "المستندات المسترجعة فعلياً وتذكر مصادرها في نهايتها."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("اسأل عن أي شيء يخص أسوان...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("جاري الاسترجاع والتوليد..."):
            context = retrieve_module.retrieve_context(query, k=3, persist=False)
            answer = prompting.generate_answer(query, context)
        st.markdown(answer)
        with st.expander("📄 الـ Chunks المسترجعة"):
            for c in context:
                st.write(f"[مصدر {c['document_id']}] ({c['category']}) {c['text']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
