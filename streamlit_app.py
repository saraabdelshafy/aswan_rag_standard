# =====================================================================
# أسوان نوباوي RAG — واجهة Streamlit الاحترافية (نسخة التحميل الكسول)
# =====================================================================
# الخط الكامل:
# 01 المستندات -> 02 التنظيف -> 03 التقطيع -> 04 التمثيل المتجهي
# -> 05 مخزن Chroma -> 06 الاسترجاع + Context Filter -> 07 Prompting
# -> 08 Ground Truth -> 09 التقييم -> عرض الإجابة مع مصادرها.
# =====================================================================
import html
import importlib
import importlib.util
import os
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
# =====================================================================
#  التحميل الكسول للوحدات — لا شيء ثقيل يعمل عند فتح الصفحة
#  (يدعم أسماء الملفات الرقمية مثل 01_documents.py)
# =====================================================================
_modules: dict = {}
_module_errors: dict = {}
def _import_module_file(name: str):
    """استيراد وحدة قد يكون اسمها رقماً (مثل 01_documents).
    أسماء الوحدات التي تبدأ برقم غير صالحة لـ importlib.import_module القياسي
    (Python يتعامل معها كأسماء معرفات غير صالحة)، لذلك نبحث عن الملف
    مباشرة في مجلد المشروع ونحمّله عبر importlib.util.
    الإصلاح: هذا هو الخطأ الأساسي في النسخة الأصلية — كان الاستيراد
    يفشل دائماً فتبقى الإحصائيات صفراً ويظهر خطأ الاسترجاع دائماً.
    """
    # 1) جرّب الاستيراد القياسي أولاً (للوحدات ذات الأسماء الصالحة)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError:
        pass
    # 2) ابحث عن ملف مطابق في مجلد المشروع أو في مجلد modules/
    base_dir = Path(__file__).resolve().parent
    candidates = [
        base_dir / f"{name}.py",
        base_dir / "modules" / f"{name}.py",
    ]
    for candidate in candidates:
        if candidate.exists():
            spec = importlib.util.spec_from_file_location(name, candidate)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[name] = module  # حتى يعمل الاستيراد المتبادل
                spec.loader.exec_module(module)
                return module
    raise ModuleNotFoundError(f"تعذّر العثور على ملف الوحدة `{name}`")
def get_module(name: str):
    """تحميل وحدة عند أول استخدام فقط، مع التقاط أي خطأ."""
    if name in _modules:
        return _modules[name]
    if name in _module_errors:
        return None
    try:
        mod = _import_module_file(name)
        _modules[name] = mod
        return mod
    except Exception as exc:  # noqa: BLE001
        _module_errors[name] = exc
        return None
def module_error(name: str) -> str:
    exc = _module_errors.get(name)
    return f"{type(exc).__name__}: {str(exc)[:220]}" if exc else ""
def reset_module_cache() -> None:
    """إعادة تعيين ذاكرة التحميل (مفيدة بعد إضافة ملفات جديدة)."""
    _modules.clear()
    _module_errors.clear()
# ================= مفاتيح الإعداد من Streamlit Secrets =================
def _apply_secrets():
    prompting = get_module("07_prompting")
    if prompting is None:
        return
    try:
        if not getattr(prompting, "OPENROUTER_API_KEY", ""):
            prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
        prompting.OPENROUTER_MODEL = st.secrets.get(
            "OPENROUTER_MODEL", getattr(prompting, "OPENROUTER_MODEL", "")
        )
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
SCENE_IMAGE = (
    "https://images.pexels.com/photos/15131486/pexels-photo-15131486.jpeg"
    "?auto=compress&cs=tinysrgb&fit=crop&h=600&w=1200"
)
WELCOME_MESSAGE = (
    "بوابة المعرفة التراثية لأهل أسوان جاهزة لاستقبال استفسارك.\n\n"
    "اكتب موضوعك (عادات، مأكولات، تراث، معالم...) وسيتم الرد بناءً على "
    "المستندات الموثّقة في قاعدة المعرفة مع ذكر المصادر في نهاية الإجابة."
)
DEFAULT_CATEGORIES = ["عادات", "مأكولات", "ملابس", "مهرجانات", "عمارة", "حرف", "تراث", "سياحة"]
# ================= إحصائيات خفيفة (وحدات البيانات فقط) =================
def _doc_stats():
    docs_mod = get_module("01_documents")
    if docs_mod is None:
        return 0, list(DEFAULT_CATEGORIES)
    try:
        docs = docs_mod.get_documents()
        cats = sorted(
            {
                (d.get("category") if isinstance(d, dict) else getattr(d, "category", "عام"))
                for d in docs
            }
        )
        return len(docs), (cats or ["عام"])
    except Exception:
        return 0, ["عام"]
def _gt_list():
    gt_mod = get_module("08_ground_truth")
    if gt_mod is None:
        return []
    try:
        return gt_mod.get_ground_truth() or []
    except Exception:
        return []
DOC_COUNT, CATEGORIES = _doc_stats()
GROUND_TRUTH = _gt_list()
GT_COUNT = len(GROUND_TRUTH)
# =====================================================================
#  التنسيق العام (CSS) — هوية بصرية داكنة نوبية
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;800&display=swap');
html, body, .stApp { font-family: 'Cairo', sans-serif; }
.stApp { background: #0a0e14; color: #e5e7eb; }
html, body, .stApp, [data-testid="stAppViewContainer"] { direction: rtl; text-align: right; }
.block-container { max-width: 1180px; padding-top: 1.2rem; padding-bottom: 2rem; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: rgba(10,14,20,0.6); backdrop-filter: blur(10px); }
/* ============ الشريط الجانبي ============ */
[data-testid="stSidebar"] { background: #0f141b; border-left: 1px solid rgba(255,255,255,0.08); }
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
    background: transparent; color: #9ca3af; font-weight: 700; font-size: 15px;
    border-radius: 12px 12px 0 0; padding: 10px 20px;
}
[data-baseweb="tab"]:hover { color: #ffffff; }
[data-baseweb="tab"][aria-selected="true"] {
    color: #fbbf24; background: rgba(251,191,36,0.08); border-bottom: 3px solid #f59e0b;
}
[data-baseweb="tab-highlight"] { display: none; }
/* ============ رسائل الشات ============ */
[data-testid="stChatMessage"] {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px; padding: 14px 18px; margin-bottom: 10px;
    direction: rtl; text-align: right;
}
[data-testid="stChatMessage"] p { direction: rtl; text-align: right; }
[data-testid="chatAvatarIcon"] {
    background: linear-gradient(135deg, #f59e0b, #f97316) !important;
    color: #0a0e14 !important; border-radius: 10px !important;
}
[data-testid="stChatInput"] {
    border: 1px solid rgba(255,255,255,0.15); border-radius: 16px;
    background: rgba(0,0,0,0.35); direction: rtl;
}
[data-testid="stChatInput"] textarea { font-family: 'Cairo', sans-serif; }
/* ============ الأزرار ============ */
.stButton > button, .stDownloadButton > button {
    font-family: 'Cairo', sans-serif; font-weight: 700; border-radius: 12px;
    background: linear-gradient(90deg, #f59e0b, #f97316); color: #0a0e14; border: none;
    padding: 10px 24px; height: auto; box-shadow: 0 8px 20px rgba(249,115,22,0.25);
}
.stButton > button:hover { background: linear-gradient(90deg, #fbbf24, #fb923c); color: #0a0e14; }
/* ============ البطاقات والمقاييس ============ */
.stExpander {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 16px !important; overflow: hidden;
}
.stExpander summary, .stExpander details { direction: rtl; text-align: right; }
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 14px;
}
[data-testid="stMetricValue"] { color: #fbbf24; font-weight: 800; }
[data-testid="stMetricLabel"] { color: #9ca3af; }
[data-testid="stMetricDelta"] { color: #34d399; }
[data-testid="stNumberInput"] input { color: #e5e7eb; background: #0f141b; }
.stAlert, .stSuccess { border-radius: 16px; }
.stImage img { border-radius: 20px; border: 1px solid rgba(255,255,255,0.1); }
hr { border-color: rgba(255,255,255,0.08); }
/* ============ الهيرو ============ */
.aswan-hero {
    position: relative; border-radius: 24px; overflow: hidden; margin-bottom: 20px;
    background:
        linear-gradient(180deg, rgba(10,14,20,0.50) 0%, rgba(10,14,20,0.85) 100%),
        url('https://images.pexels.com/photos/18934581/pexels-photo-18934581.jpeg?auto=compress&cs=tinysrgb&fit=crop&h=900&w=1600')
        center/cover no-repeat;
    min-height: 320px; display: flex; align-items: flex-end;
}
.aswan-hero-content { padding: 30px 34px; max-width: 720px; }
.aswan-hero-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(255,255,255,0.12); border: 1px solid rgba(255,255,255,0.25);
    border-radius: 999px; padding: 6px 14px; font-size: 12px; font-weight: 700;
    color: #fcd34d; backdrop-filter: blur(8px);
}
.aswan-hero-title { font-size: 36px; font-weight: 800; line-height: 1.3; color: #fff; margin: 14px 0 10px 0; }
.aswan-hero-title span {
    background: linear-gradient(90deg, #fbbf24, #fb923c);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.aswan-hero-desc { font-size: 15px; color: rgba(255,255,255,0.82); line-height: 1.9; }
/* ============ الإحصائيات ============ */
.aswan-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 22px; }
.aswan-stat {
    background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 14px 16px; display: flex; align-items: center; gap: 12px;
}
.aswan-stat-icon {
    width: 44px; height: 44px; border-radius: 12px; background: rgba(251,191,36,0.12);
    border: 1px solid rgba(255,255,255,0.1); display: grid; place-items: center; font-size: 20px;
}
.aswan-stat-value { font-size: 22px; font-weight: 800; color: #fff; line-height: 1.2; }
.aswan-stat-label { font-size: 11px; color: #9ca3af; }
/* ============ بطاقة المصدر ============ */
.chunk-card {
    background: rgba(0,0,0,0.30); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px; padding: 12px 14px; margin-bottom: 10px;
    direction: rtl; text-align: right;
}
.chunk-card-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; gap: 10px; flex-wrap: wrap; }
.chunk-card-title { color: #fbbf24; font-weight: 700; font-size: 13px; }
.chunk-card-id { color: #9ca3af; font-weight: 400; }
.chunk-card-meta { color: #9ca3af; font-size: 11px; }
.chunk-card-text { color: #d1d5db; font-size: 12.5px; line-height: 1.9; white-space: pre-wrap; }
/* ============ رأس التقييم ============ */
.eval-header {
    display: flex; align-items: center; gap: 14px;
    background: linear-gradient(90deg, rgba(251,191,36,0.10), rgba(255,255,255,0.02));
    border: 1px solid rgba(255,255,255,0.1); border-radius: 20px;
    padding: 18px 22px; margin-bottom: 16px;
}
.eval-header-icon {
    width: 52px; height: 52px; border-radius: 14px;
    background: linear-gradient(135deg, #f59e0b, #f97316);
    display: grid; place-items: center; font-size: 24px; flex-shrink: 0;
}
.eval-header-title { font-size: 20px; font-weight: 800; color: #fff; }
.eval-header-sub { font-size: 13px; color: #9ca3af; margin-top: 2px; }
/* ============ الشرائح والأشرطة ============ */
.cat-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.cat-chip {
    font-size: 11px; padding: 3px 12px; border-radius: 999px;
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1); color: #d1d5db;
}
.gt-bar-row { margin-bottom: 10px; }
.gt-bar-head { display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px; }
.gt-bar-head span:first-child { color: #d1d5db; }
.gt-bar-head span:last-child { color: #fbbf24; font-weight: 700; }
.gt-bar-track { height: 6px; border-radius: 999px; background: rgba(255,255,255,0.08); overflow: hidden; }
.gt-bar-fill { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #fbbf24, #f97316); }
/* ============ حالة النظام ============ */
.sys-dot { display: inline-block; width: 8px; height: 8px; border-radius: 999px; margin-left: 6px; }
.sys-ok { background: #34d399; }
.sys-err { background: #f87171; }
/* ============ التذييل ============ */
.aswan-footer { margin-top: 40px; border-top: 1px solid rgba(255,255,255,0.08); padding: 22px 0 8px 0; text-align: center; }
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
_apply_secrets()
# =====================================================================
#  مكوّنات HTML مساعدة
# =====================================================================
def hero_html() -> str:
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
    doc_id = chunk.get("document_id", "؟")
    category = chunk.get("category", "")
    distance = chunk.get("distance", 0.0)
    # إصلاح: تحويل فواصل الأسطر إلى <br> حتى لا يظهر النص في سطر واحد
    text = html.escape(str(chunk.get("text", ""))).replace("\n", "<br>")
    title = chunk.get("title") or f"مصدر {doc_id}"
    try:
        dist_txt = f"{float(distance):.3f}"
    except Exception:
        dist_txt = str(distance)
    return (
        f'<div class="chunk-card"><div class="chunk-card-head">'
        f'<span class="chunk-card-title">◾ {html.escape(str(title))} '
        f'<span class="chunk-card-id">[{html.escape(str(doc_id))}]</span></span>'
        f'<span class="chunk-card-meta">distance {dist_txt} • {html.escape(str(category))}</span>'
        f'</div><div class="chunk-card-text">{text}</div></div>'
    )
def categories_chips_html() -> str:
    chips = "".join(f'<span class="cat-chip">{html.escape(str(c))}</span>' for c in CATEGORIES)
    return f'<div class="cat-chips">{chips}</div>'
def gt_distribution_html(gt) -> str:
    rows = []
    for cat in CATEGORIES:
        count = sum(1 for g in gt if g.get("category") == cat)
        pct = (count / len(gt)) * 100 if gt else 0
        rows.append(
            f'<div class="gt-bar-row"><div class="gt-bar-head">'
            f'<span>{html.escape(str(cat))}</span><span>{count}</span></div>'
            f'<div class="gt-bar-track"><div class="gt-bar-fill" style="width:{pct:.0f}%"></div></div></div>'
        )
    return "".join(rows)
def footer_html() -> str:
    return """
    <div class="aswan-footer">
        <div class="aswan-footer-main">© 2025 أسوان نوباوي RAG • مشروع توثيق تراث النوبة</div>
        <div class="aswan-footer-sub">
            الخط الكامل: 01 المستندات ← 02 التنظيف ← 03 التقطيع ← 04 التمثيل المتجهي ←
            05 مخزن Chroma ← 06 الاسترجاع + Context Filter ← 07 Prompting ←
            08 Ground Truth ← 09 التقييم
        </div>
    </div>
    """
def sanitize_chunks(chunks) -> list:
    """قيم بسيطة فقط قبل التخزين في Session State."""
    safe = []
    for c in chunks or []:
        try:
            safe.append({
                "document_id": str(c.get("document_id", "؟")),
                "category": str(c.get("category", "")),
                "distance": float(c.get("distance", 0.0)),
                "text": str(c.get("text", "")),
                "title": str(c.get("title", "")),
            })
        except Exception:
            continue
    return safe
def ensure_messages_initialized() -> None:
    """تهيئة رسائل الجلسة بأمان (إصلاح: لا KeyError عند أول تشغيل)."""
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "chunks": None}
        ]
# =====================================================================
#  الشريط الجانبي
# =====================================================================
with st.sidebar:
    st.markdown("### ⚙️ إعدادات الاسترجاع")
    k_value = st.slider("عدد المستندات المسترجعة (k)", 1, 6, 3)
    max_dist = st.slider("أقصى مسافة مسموح بها (Context Filter)", 0.1, 1.5, 0.8, 0.05)
    dedup = st.checkbox("إزالة التكرار على مستوى المستند", value=True)
    st.divider()
    st.markdown("#### 🗂️ فئات المعرفة")
    st.markdown(categories_chips_html(), unsafe_allow_html=True)
    st.divider()
    st.markdown("#### 🩺 حالة وحدات النظام")
    _check = [
        ("01_documents", True), ("06_retrieve_context", False),
        ("07_prompting", False), ("08_ground_truth", True), ("09_evaluation", False),
    ]
    for _name, _light in _check:
        if _name in _module_errors:
            st.markdown(
                f'<span class="sys-dot sys-err"></span> <code>{_name}</code> — فشل التحميل',
                unsafe_allow_html=True,
            )
        elif _light and _name in _modules:
            st.markdown(
                f'<span class="sys-dot sys-ok"></span> <code>{_name}</code> — جاهزة',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<span class="sys-dot" style="background:#6b7280"></span> '
                f'<code>{_name}</code> — عند الطلب',
                unsafe_allow_html=True,
            )
    if _module_errors:
        with st.expander("تفاصيل الأعطال"):
            for _n in _module_errors:
                st.caption(f"`{_n}` → {module_error(_n)}")
    st.divider()
    if st.button("🔄 محادثة جديدة"):
        st.session_state.messages = [
            {"role": "assistant", "content": WELCOME_MESSAGE, "chunks": None}
        ]
        st.rerun()
    with st.expander("ℹ️ عن المنصة"):
        st.markdown(
            f"قاعدة المعرفة تضم **{DOC_COUNT}** مستنداً عن عادات وتراث أسوان. "
            "كل إجابة تُبنى فقط من المستندات المسترجعة بعد تمريرها على "
            "Context Filter مع ذكر مصادرها في نهاية الإجابة."
        )
# =====================================================================
#  الواجهة الرئيسية — تُرسم دائماً بغضّ النظر عن حالة الوحدات
# =====================================================================
ensure_messages_initialized()
st.markdown(hero_html(), unsafe_allow_html=True)
st.markdown(stats_html(), unsafe_allow_html=True)
tab_chat, tab_eval = st.tabs(["💬 الاستفسار الذكي", "📊 التقييم ونتائج الأداء"])
# ================= تبويب الاستفسار الذكي =================
with tab_chat:
    st.image(SCENE_IMAGE, caption="مراكب النيل في أسوان — نافذة على التراث النوبي")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🏛️" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg.get("chunks"):
                with st.expander(f"📚 المصادر المسترجعة ({len(msg['chunks'])})"):
                    for c in msg["chunks"]:
                        st.markdown(chunk_card_html(c), unsafe_allow_html=True)
    query = st.chat_input("اكتب موضوعاً عن عادات وتقاليد أهل أسوان...")
    if query:
        st.session_state.messages.append({"role": "user", "content": query, "chunks": None})
        with st.chat_message("user", avatar="👤"):
            st.markdown(query)
        with st.chat_message("assistant", avatar="🏛️"):
            answer = ""
            context = []
            with st.spinner("جارٍ الاسترجاع والفلترة السياقية..."):
                retrieve_module = get_module("06_retrieve_context")
                if retrieve_module is None:
                    answer = (
                        "⚠️ تعذّر تحميل وحدة الاسترجاع `06_retrieve_context`.\n\n"
                        f"التفاصيل: {module_error('06_retrieve_context')}\n\n"
                        "غالباً السبب: مكتبة ناقصة في `requirements.txt` أو ملف بيانات "
                        "غير مرفوع في المستودع."
                    )
                else:
                    try:
                        context = retrieve_module.retrieve_context(
                            query, k=k_value, persist=False,
                            max_distance=max_dist, deduplicate_by_document=dedup,
                        )
                    except Exception as exc:  # noqa: BLE001
                        answer = f"⚠️ حدث خطأ أثناء الاسترجاع: {type(exc).__name__}: {exc}"
                prompting = None
                if context:
                    prompting = get_module("07_prompting")
                    if prompting is None:
                        answer = f"⚠️ تعذّر تحميل `07_prompting`: {module_error('07_prompting')}"
                    else:
                        try:
                            answer = prompting.generate_answer(query, context)
                        except Exception as exc:  # noqa: BLE001
                            answer = f"⚠️ خطأ أثناء التوليد: {type(exc).__name__}: {exc}"
                elif not answer:
                    answer = (
                        "⚠️ لم يتم العثور على مستندات قريبة كفاية بعد تطبيق "
                        "Context Filter. جرّب تخفيف max_distance أو إعادة صياغة السؤال."
                    )
            # نعرض صندوق المصادر فقط لو فعلاً فيه context ومفيش رفض من النموذج.
            # (وجود context لوحده مش كافي: ممكن الـ chunks تكون اترجعت لكنها
            # غير كافية للإجابة، فيرفض النموذج، وحينها لا تُعتبر "مصادر مستخدمة".)
            show_sources = bool(context) and not (prompting is not None and prompting.is_refusal(answer))
            st.markdown(answer)
            if show_sources:
                with st.expander(f"📚 المصادر المسترجعة ({len(context)})"):
                    for c in context:
                        st.markdown(chunk_card_html(c), unsafe_allow_html=True)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "chunks": sanitize_chunks(context) if show_sources else None,
            }
        )
# ================= تبويب التقييم =================
with tab_eval:
    st.markdown(
        """
        <div class="eval-header">
            <div class="eval-header-icon">📊</div>
            <div>
                <div class="eval-header-title">مقياس أداء النظام</div>
                <div class="eval-header-sub">
                    تقييم جودة الاسترجاع مقارنةً بمجموعة أسئلة مرجعية (Ground Truth)
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"مجموعة التقييم تحتوي على **{GT_COUNT}** سؤال مرجعي، لكل سؤال المستندات "
        "المتوقع استرجاعها والكلمات المفتاحية المتوقع ظهورها في الإجابة."
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        k_eval = st.number_input("k للتقييم", min_value=1, max_value=10, value=3)
    with col2:
        max_dist_eval = st.number_input(
            "max_distance للفلترة", min_value=0.1, max_value=2.0, value=0.8, step=0.05
        )
    with col3:
        st.write("")
        st.write("")
        run_eval = st.button("▶ تشغيل التقييم الشامل")
    if run_eval:
        evaluation_module = get_module("09_evaluation")
        if evaluation_module is None:
            st.error(
                f"⚠️ تعذّر تحميل `09_evaluation`: {module_error('09_evaluation')}"
            )
        else:
            with st.spinner("جارٍ تشغيل التقييم على كل الأسئلة المرجعية..."):
                try:
                    raw_summary, _ = evaluation_module.evaluate_retrieval(k=k_eval, use_filter=False)
                    filtered_summary, filtered_results = evaluation_module.evaluate_retrieval(
                        k=k_eval, use_filter=True, max_distance=max_dist_eval
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"حدث خطأ أثناء التقييم: {type(exc).__name__}: {exc}")
                    st.stop()
            st.success("✅ اكتمل تشغيل التقييم بنجاح")
            st.markdown("### 📈 المقارنة البصرية")
            chart = pd.DataFrame(
                {
                    "بدون فلترة": [
                        raw_summary.get("avg_precision@k", 0), raw_summary.get("avg_recall@k", 0),
                        raw_summary.get("avg_f1@k", 0), raw_summary.get("mrr", 0),
                        raw_summary.get("hit_rate", 0),
                    ],
                    "مع Context Filter": [
                        filtered_summary.get("avg_precision@k", 0), filtered_summary.get("avg_recall@k", 0),
                        filtered_summary.get("avg_f1@k", 0), filtered_summary.get("mrr", 0),
                        filtered_summary.get("hit_rate", 0),
                    ],
                },
                index=["Precision@k", "Recall@k", "F1@k", "MRR", "Hit Rate"],
            )
            st.bar_chart(chart, height=320)
            metric_keys = [
                ("avg_precision@k", "Precision@k"), ("avg_recall@k", "Recall@k"),
                ("avg_f1@k", "F1@k"), ("mrr", "MRR"), ("hit_rate", "Hit Rate"),
            ]
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### أساسي (دون فلترة سياقية)")
                for col, (key, label) in zip(st.columns(5), metric_keys):
                    with col:
                        st.metric(label, f"{raw_summary.get(key, 0):.3f}")
            with col_b:
                st.markdown("#### مُفلتر (مع Context Filter)")
                for col, (key, label) in zip(st.columns(5), metric_keys):
                    with col:
                        st.metric(
                            label,
                            f"{filtered_summary.get(key, 0):.3f}",
                            delta=f"{(filtered_summary.get(key, 0) - raw_summary.get(key, 0)):+.3f}",
                        )
            st.markdown("### 📚 توزيع أسئلة الاختبار المرجعي حسب الفئة")
            st.markdown(gt_distribution_html(GROUND_TRUTH), unsafe_allow_html=True)
            st.markdown("### 📋 تفاصيل كل سؤال (مع الفلترة)")
            for r in filtered_results:
                status = "✅" if r.get("reciprocal_rank", 0) > 0 else "❌"
                with st.expander(f"{status} {r.get('question', '')}"):
                    col_i, col_j = st.columns(2)
                    with col_i:
                        st.markdown(f"**المتوقع:** {r.get('expected', '')}")
                    with col_j:
                        st.markdown(f"**المسترجع:** {r.get('retrieved', '')}")
                    st.markdown(
                        f"**Precision:** {r.get('precision', 0):.2f} | "
                        f"**Recall:** {r.get('recall', 0):.2f} | "
                        f"**F1:** {r.get('f1', 0):.2f}"
                    )
    with st.expander("📖 استعرض بنك أسئلة Ground Truth"):
        for item in GROUND_TRUTH:
            st.markdown(
                f"**{item.get('id', '؟')}.** {item.get('question', '')} — "
                f"docs: {item.get('expected_document_ids', [])} | "
                f"الفئة: {item.get('category', '')}"
            )
st.markdown(footer_html(), unsafe_allow_html=True)
