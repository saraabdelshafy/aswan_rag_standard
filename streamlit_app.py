"""
=====================================================================
 أسوان نوباوي RAG — واجهة Streamlit الاحترافية (نسخة قابلة للعرض التجاري)
=====================================================================
نسخة مطوّرة من واجهة المشروع بهوية بصرية احترافية:
- صورة حقيقية لمعبد فيلة (أسوان) في الواجهة الرئيسية
- نظام ألوان نوبي (كهرماني/برتقالي) + خط Cairo + دعم كامل للـ RTL
- تبويبان: «الاستفسار الذكي» + «التقييم ونتائج الأداء»
- كل الإجابات مبنية من المستندات الموثّقة مع توثيق المصادر

الخط الكامل:
المستندات (01) -> التنظيف (02) -> التقطيع (03) -> التمثيل المتجهي (04)
-> مخزن Chroma (05) -> الاسترجاع + Context Filter (06) -> الـ Prompting (07)
-> Ground Truth (08) -> التقييم (09) -> عرض الإجابة مع مصادرها.

طريقة التشغيل:
    streamlit run streamlit_app.py
=====================================================================
"""

import importlib
import html

import pandas as pd
import streamlit as st

# ================= استيراد مراحل الـ pipeline =================
documents_module = importlib.import_module("01_documents")
preprocessing_module = importlib.import_module("02_preprocessing")
chunking_module = importlib.import_module("03_chunking")
vector_module = importlib.import_module("04_vector_representation")
store_module = importlib.import_module("05_create_chroma_store")
retrieve_module = importlib.import_module("06_retrieve_context")
prompting = importlib.import_module("07_prompting")
ground_truth_module = importlib.import_module("08_ground_truth")
evaluation_module = importlib.import_module("09_evaluation")

# ================= مفاتيح الإعداد من Streamlit Secrets =================
try:
    if not prompting.OPENROUTER_API_KEY:
        prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    prompting.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", prompting.OPENROUTER_MODEL)
except Exception:
    pass

# ================= إعداد الصفحة =================
st.set_page_config(
    page_title="أسوان نوباوي | بوابة تراث أهل أسوان",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================= ثوابت عامة =================
HERO_IMAGE = (
    "https://images.pexels.com/photos/18934581/pexels-photo-18934581.jpeg"
    "?auto=compress&cs=tinysrgb&fit=crop&h=900&w=1600"
)
SCENE_IMAGE = (
    "https://images.pexels.com/photos/15131486/pexels-photo-15131486.jpeg"
    "?auto=compress&cs=tinysrgb&fit=crop&h=600&w=1200"
)

WELCOME_MESSAGE = (
    "بوابة المعرفة التراثية لأهل أسوان جاهزة لاستقبال استفسارك.\n\n"
    "اكتب موضوعك (عادات، مأكولات، تراث، معالم...) وسيتم الرد بناءً على "
    "المستندات الموثّقة في قاعدة المعرفة مع ذكر المصادر في نهاية الإجابة."
)

# ================= إحصائيات المشروع (بأمان) =================
try:
    _docs = documents_module.get_documents()
    DOC_COUNT = len(_docs)
    CATEGORIES = sorted(
        {
            (d.get("category") if isinstance(d, dict) else getattr(d, "category", "عام"))
            for d in _docs
        }
    )
except Exception:
    DOC_COUNT = 0
    CATEGORIES = ["عادات", "مأكولات", "ملابس", "مهرجانات", "عمارة", "حرف", "تراث", "سياحة"]

try:
    GT_COUNT = len(ground_truth_module.get_ground_truth())
except Exception:
    GT_COUNT = 0

# =====================================================================
#  التنسيق العام (CSS) — هوية بصرية داكنة نوبية
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;800&display=swap');

html, body, .stApp, [class*="css"] { font-family: 'Cairo', sans-serif; }

.stApp { background: #0a0e14; color: #e5e7eb; }

html, body, .stApp, [data-testid="stAppViewContainer"] { direction: rtl; text-align: right; }

.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 2rem; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(10,14,20,0.6); backdrop-filter: blur(10px); }

/* ============ الشريط الجانبي ============ */
[data-testid="stSidebar"] {
    background: #0f141b;
    border-left: 1px solid rgba(255,255,255,0.08);
}
[data-testid="stSidebar"] .block-container { padding-top: 1.2rem; }
[data-testid="stSidebar"] h3 { color: #fbbf24; font-weight: 800; }
[data-testid="stSidebar"] h4 { color: #e5e7eb; font-weight: 700; }
[data-testid="stSidebar"] .stSlider input[type="range"] { accent-color: #f59e0b; }
[data-testid="stSidebar"] .stCheckbox input[type="checkbox"] { accent-color: #f59e0b; }
[data-testid="stSidebar"] .stButton > button { width: 100%; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.08); }

/* ============ التبويبات ============ */
[data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid rgba(255,255,255,0.1); }
[data-baseweb="tab"] {
    background: transparent;
    color: #9ca3af;
    font-weight: 700;
    font-size: 15px;
    border-radius: 12px 12px 0 0;
    padding: 10px 20px;
}
[data-baseweb="tab"]:hover { color: #ffffff; }
[data-baseweb="tab"][aria-selected="true"] {
    color: #fbbf24;
    background: rgba(251,191,36,0.08);
    border-bottom: 3px solid #f59e0b;
}
[data-baseweb="tab-highlight"] { display: none; }

/* ============ رسائل الشات ============ */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
[data-testid="chatAvatarIcon"] {
    background: linear-gradient(135deg, #f59e0b, #f97316) !important;
    color: #0a0e14 !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] {
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 16px;
    background: rgba(0,0,0,0.35);
    direction: rtl;
}
[data-testid="stChatInput"] textarea { font-family: 'Cairo', sans-serif; }

/* ============ الأزرار ============ */
.stButton > button, .stDownloadButton > button {
    font-family: 'Cairo', sans-serif;
    font-weight: 700;
    border-radius: 12px;
    background: linear-gradient(90deg, #f59e0b, #f97316);
    color: #0a0e14;
    border: none;
    padding: 10px 24px;
    height: auto;
    box-shadow: 0 8px 20px rgba(249,115,22,0.25);
}
.stButton > button:hover {
    background: linear-gradient(90deg, #fbbf24, #fb923c);
    color: #0a0e14;
}

/* ============ البطاقات والمقاييس ============ */
.stExpander {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important;
    overflow: hidden;
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 14px;
}
[data-testid="stMetricValue"] { color: #fbbf24; font-weight: 800; }
[data-testid="stMetricLabel"] { color: #9ca3af; }
[data-testid="stMetricDelta"] { color: #34d399; }
[data-testid="stNumberInput"] input { color: #e5e7eb; background: #0f141b; }
[data-testid="stNumberInput"] button { color: #fbbf24; }

.stAlert, .stSuccess { border-radius: 16px; }
.stImage img { border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); }
hr { border-color: rgba(255,255,255,0.08); }

/* ============ الهيرو (الصورة الرئيسية) ============ */
.aswan-hero {
    position: relative;
    border-radius: 24px;
    overflow: hidden;
    margin-bottom: 20px;
    background:
        linear-gradient(180deg, rgba(10,14,20,0.50) 0%, rgba(10,14,20,0.85) 100%),
        url('https://images.pexels.com/photos/18934581/pexels-photo-18934581.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=900&w=1600')
        center/cover no-repeat;
    min-height: 320px;
    display: flex;
    align-items: flex-end;
}
.aswan-hero-content { padding: 30px 34px; max-width: 720px; }
.aswan-hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px;
    padding: 6px 14px;
    font-size: 12px;
    font-weight: 700;
    color: #fcd34d;
    backdrop-filter: blur(8px);
}
.aswan-hero-title {
    font-size: 36px;
    font-weight: 800;
    line-height: 1.3;
    color: #ffffff;
    margin: 14px 0 10px 0;
}
.aswan-hero-title span {
    background: linear-gradient(90deg, #fbbf24, #fb923c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.aswan-hero-desc {
    font-size: 15px;
    color: rgba(255,255,255,0.82);
    line-height: 1.9;
}

/* ============ شريط الإحصائيات ============ */
.aswan-stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 22px;
}
.aswan-stat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 14px 16px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.aswan-stat-icon {
    width: 44px; height: 44px;
    border-radius: 12px;
    background: rgba(251,191,36,0.12);
    border: 1px solid rgba(255,255,255,0.1);
    display: grid; place-items: center;
    font-size: 20px;
}
.aswan-stat-value { font-size: 22px; font-weight: 800; color: #ffffff; line-height: 1.2; }
.aswan-stat-label { font-size: 11px; color: #9ca3af; }

/* ============ بطاقة المصدر (Chunk) ============ */
.chunk-card {
    background: rgba(0,0,0,0.30);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 12px 14px;
    margin-bottom: 10px;
}
.chunk-card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 6px;
    gap: 10px;
    flex-wrap: wrap;
}
.chunk-card-title { color: #fbbf24; font-weight: 700; font-size: 13px; }
.chunk-card-id { color: #9ca3af; font-weight: 400; }
.chunk-card-meta { color: #9ca3af; font-size: 11px; }
.chunk-card-text { color: #d1d5db; font-size: 12.5px; line-height: 1.9; }

/* ============ رأس صفحة التقييم ============ */
.eval-header {
    display: flex;
    align-items: center;
    gap: 14px;
    background: linear-gradient(90deg, rgba(251,191,36,0.10), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px;
    padding: 18px 22px;
    margin-bottom: 16px;
}
.eval-header-icon {
    width: 52px; height: 52px;
    border-radius: 14px;
    background: linear-gradient(135deg, #f59e0b, #f97316);
    display: grid; place-items: center;
    font-size: 24px;
    flex-shrink: 0;
}
.eval-header-title { font-size: 20px; font-weight: 800; color: #ffffff; }
.eval-header-sub { font-size: 13px; color: #9ca3af; margin-top: 2px; }

/* ============ فئات المعرفة (Chips) ============ */
.cat-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.cat-chip {
    font-size: 11px;
    padding: 3px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    color: #d1d5db;
}

/* ============ توزيع الفئات في التقييم ============ */
.gt-bar-row { margin-bottom: 10px; }
.gt-bar-head {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    margin-bottom: 4px;
}
.gt-bar-head span:first-child { color: #d1d5db; }
.gt-bar-head span:last-child { color: #fbbf24; font-weight: 700; }
.gt-bar-track {
    height: 6px;
    border-radius: 999px;
    background: rgba(255,255,255,0.08);
    overflow: hidden;
}
.gt-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #fbbf24, #f97316);
}

/* ============ التذييل ============ */
.aswan-footer {
    margin-top: 40px;
    border-top: 1px solid rgba(255,255,255,0.08);
    padding: 22px 0 8px 0;
    text-align: center;
}
.aswan-footer-main { color: #9ca3af; font-size: 13px; font-weight: 600; }
.aswan-footer-sub { color: #6b7280; font-size: 11px; margin-top: 6px; line-height: 1.8; }

@media (max-width: 900px) {
    .aswan-stats { grid-template-columns: repeat(2, 1fr); }
    .aswan-hero-title { font-size: 26px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
#  مكوّنات HTML مساعدة
# =====================================================================

def hero_html() -> str:
    """الواجهة الرئيسية بصورة معبد فيلة."""
    return """
    <div class="aswan-hero">
        <div class="aswan-hero-content">
            <div class="aswan-hero-badge">📍 عاصمة النوبة • بوابة حضارة وادي النيل</div>
            <div class="aswan-hero-title">بوابة تراث أسوان<br>
                <span>استكشف عادات وتقاليد أهل النوبة</span>
            </div>
            <div class="aswan-hero-desc">
                منصة معرفة ذكية توثّق العادات والمأكولات والتراث والمعالم الشهيرة لأسوان
                عبر نظام استرجاع وتوليد معزز بالمعرفة (RAG)، مع توثيق مصدر كل معلومة.
            </div>
        </div>
    </div>
    """


def stats_html() -> str:
    """شريط الإحصائيات."""
    cards = [
        ("📄", str(DOC_COUNT), "مستند موثّق"),
        ("🗂️", str(len(CATEGORIES)), "فئة تراثية"),
        ("❓", str(GT_COUNT), "سؤال معياري"),
        ("🛡️", "Context", "استرجاع مُفلتر"),
    ]
    inner = "".join(
        f'<div class="aswan-stat"><div class="aswan-stat-icon">{icon}</div>'
        f'<div><div class="aswan-stat-value">{value}</div>'
        f'<div class="aswan-stat-label">{label}</div></div></div>'
        for icon, value, label in cards
    )
    return f'<div class="aswan-stats">{inner}</div>'


def chunk_card_html(chunk: dict) -> str:
    """بطاقة مصدر مسترجَع (Chunk) بتنسيق احترافي."""
    doc_id = chunk.get("document_id", "؟")
    category = chunk.get("category", "")
    distance = chunk.get("distance", 0.0)
    text = html.escape
