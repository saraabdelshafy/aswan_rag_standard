"""
streamlit_app.py
واجهة Streamlit النهائية لمشروع أسوان RAG.

تجمع كل مراحل الخط: المستندات (01) -> التنظيف (02) -> التقطيع (03)
-> التمثيل المتجهي (04) -> مخزن Chroma (05) -> الاسترجاع (06) -> الـ Prompting (07)
-> عرض الإجابة مع مصادرها في واجهة شات تفاعلية بألوان نوبية زاهية.
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

# ---------------------------------------------------------------------------
# إعداد الصفحة + هوية بصرية نوبية زاهية (تركواز + فوشيا + أصفر + برتقالي)
# ---------------------------------------------------------------------------
st.set_page_config(page_title="أسوان RAG", page_icon="", layout="centered")

TURQUOISE = "#00A9A5"
MAGENTA = "#D6336C"
YELLOW = "#F4B400"
ORANGE = "#F2703C"
PURPLE = "#6C3483"
BG = "#FFF9EF"
CARD = "#FFFFFF"
TEXT = "#2B2420"
MUTED = "#8A7F6E"

ACCENT_CYCLE = [TURQUOISE, MAGENTA, YELLOW, ORANGE, PURPLE]

st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@600;700;800;900&family=Tajawal:wght@400;500;700&display=swap');

    html, body, .stApp {{
        direction: rtl; text-align: right;
        background: {BG}; font-family: 'Tajawal', sans-serif; color: {TEXT};
    }}
    h1, h2, h3, .display-font {{ font-family: 'Cairo', sans-serif; }}
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{ height: 0; visibility: hidden; }}
    .block-container {{ padding-top: 3.5rem; max-width: 760px; }}

    /* هيدر متدرّج بألوان نوبية */
    .app-header {{
        background: linear-gradient(120deg, {TURQUOISE} 0%, {PURPLE} 55%, {MAGENTA} 100%);
        border-radius: 20px; padding: 1.4rem 1.6rem; margin-bottom: 1.3rem;
        display: flex; align-items: center; gap: 1rem;
        box-shadow: 0 8px 20px rgba(108,52,131,0.25);
    }}
    .app-logo {{
        width: 56px; height: 56px; border-radius: 16px; background: rgba(255,255,255,0.22);
        display: flex; align-items: center; justify-content: center; font-size: 28px;
        border: 2px solid rgba(255,255,255,0.4);
    }}
    .app-title {{ font-family: 'Cairo', sans-serif; font-weight: 900; font-size: 1.6rem; color: #FFFFFF; margin: 0; }}
    .app-subtitle {{ font-size: 0.85rem; color: rgba(255,255,255,0.92); margin: 0; }}

    /* شريط أصفر/برتقالي زخرفي تحت الهيدر (زي زخارف البيوت النوبية) */
    .pattern-strip {{
        height: 6px; border-radius: 6px; margin-bottom: 1.4rem;
        background: repeating-linear-gradient(90deg, {YELLOW} 0 20px, {ORANGE} 20px 40px, {MAGENTA} 40px 60px, {TURQUOISE} 60px 80px);
    }}

    /* كروت الأسئلة الجاهزة بألوان دايرة */
    div[data-testid="stButton"] > button {{
        border-radius: 14px !important; font-family: 'Tajawal', sans-serif !important;
        font-weight: 700 !important; padding: 0.85rem 1rem !important;
        transition: transform 0.15s ease;
        background: {CARD} !important; color: {TEXT} !important;
        border: 1px solid #E9D9C3 !important;
    }}
    div[data-testid="stButton"] > button * {{ color: {TEXT} !important; }}
    div[data-testid="stButton"] > button:hover {{
        transform: translateY(-2px) scale(1.01);
        border-color: {TURQUOISE} !important; color: {PURPLE} !important;
    }}
    div[data-testid="stButton"] > button:hover * {{ color: {PURPLE} !important; }}

    /* فقاعات الشات */
    div[data-testid="stChatMessage"] {{
        background: {CARD} !important; border-radius: 16px !important;
        padding: 0.95rem 1.15rem !important; margin-bottom: 0.8rem !important;
        border-right: 5px solid {TURQUOISE} !important;
        box-shadow: 0 2px 6px rgba(43,36,32,0.06);
    }}
    div[data-testid="stChatMessage"] * {{ color: {TEXT} !important; }}

    div[data-testid="stChatInput"] textarea {{ color: {TEXT} !important; background: {CARD} !important; }}
    div[data-testid="stChatInput"] {{ border: 2px solid {TURQUOISE}33 !important; border-radius: 16px !important; }}

    div[data-testid="stExpander"] {{
        border: 2px dashed {ORANGE}55 !important; border-radius: 14px !important; background: {CARD} !important;
    }}

    .footer-note {{ text-align: center; color: {MUTED}; font-size: 0.78rem; margin-top: 1.5rem; }}
    .footer-note b {{ color: {MAGENTA}; }}
    </style>

    <div class="app-header">
        <div class="app-logo">
            <svg width="34" height="34" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
                <circle cx="68" cy="26" r="14" fill="{YELLOW}" opacity="0.95"/>
                <path d="M50 12 L50 62 L74 62 Z" fill="#FFFFFF" opacity="0.96"/>
                <rect x="48.5" y="10" width="3" height="56" rx="1.5" fill="#FFFFFF"/>
                <path d="M20 66 Q50 54 80 66 L80 70 Q50 60 20 70 Z" fill="{TURQUOISE}"/>
                <path d="M14 76 Q50 62 86 76 L86 81 Q50 68 14 81 Z" fill="#FFFFFF" opacity="0.9"/>
                <path d="M10 88 Q50 74 90 88 L90 93 Q50 80 10 93 Z" fill="{TURQUOISE}"/>
            </svg>
        </div>
        <div>
            <p class="app-title"> أسوان نوباوي 🌴</p>
            <p class="app-subtitle">نظام استرجاع وتوليد معزز بالمعرفة عن ثقافة ومعالم أسوان</p>
        </div>
    </div>
    <div class="pattern-strip"></div>
    """,
    unsafe_allow_html=True,
)

with st.expander("ℹ️ عن المشروع"):
    st.write(
        f"قاعدة المعرفة تحتوي على {len(documents_module.get_documents())} مستنداً أصلياً عن أسوان "
        "(المعابد، الحدائق، النزهات النيلية، عادات وتقاليد النوبة). كل إجابة تُبنى فقط من "
        "المستندات المسترجعة فعلياً وتذكر مصادرها في نهايتها."
    )

# ---------------------------------------------------------------------------
# شاشة ترحيب (تظهر فقط قبل أول سؤال) + أسئلة جاهزة ملوّنة
# ---------------------------------------------------------------------------
EXAMPLES_AR = [
    "🟦 عادات وتقاليد أهل النوبة في الاحتفالات",
    "🟪 ما هي ظاهرة تعامد الشمس في أبو سمبل؟",
    "🟨 الحديقة النباتية جزيرة كتشنر",
    "🟧 مراسم ليلة الحنة عند النوبيين",
]

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None

if not st.session_state.messages:
    st.markdown(
        f'<p style="text-align:center; color:{TEXT}; font-weight:700; margin-top:0.2rem;">'
        '👋 اسأل عن معابد أسوان، حدائقها، نزهاتها النيلية، أو عادات وتقاليد النوبة</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, ex in enumerate(EXAMPLES_AR):
        with cols[i % 2]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                st.session_state.pending_query = ex.split(" ", 1)[1]
                st.rerun()

# ---------------------------------------------------------------------------
# عرض المحادثة
# ---------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🏛️" if msg["role"] == "assistant" else "🙋"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

typed_query = st.chat_input("اسأل عن أي شيء يخص أسوان...")
query = st.session_state.pending_query or typed_query
st.session_state.pending_query = None

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🙋"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🏛️"):
        with st.spinner("جاري الاسترجاع والتوليد..."):
            context = retrieve_module.retrieve_context(query, k=3, persist=False)
            answer = prompting.generate_answer(query, context)
        st.markdown(answer)
        with st.expander("📄 الـ Chunks المسترجعة"):
            for c in context:
                st.write(f"[مصدر {c['document_id']}] ({c['category']}) {c['text']}")

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.markdown(
    '<p class="footer-note">🌴 الإجابات مبنية على مصادر ثقافية موثقة · '
    'مشروع أكاديمي — استشيري مرشداً سياحياً محلياً للتفاصيل الدقيقة.</p>',
    unsafe_allow_html=True,
)
