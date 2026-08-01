"""
streamlit_app.py
واجهة Streamlit النهائية لمشروع أسوان RAG.

تجمع كل مراحل الخط: المستندات (01) -> التنظيف (02) -> التقطيع (03)
-> التمثيل المتجهي (04) -> مخزن Chroma (05) -> الاسترجاع + Context Filter (06)
-> الـ Prompting (07) -> Ground Truth (08) -> التقييم (09)
-> عرض الإجابة مع مصادرها في واجهة شات تفاعلية + صفحة تقييم.
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
ground_truth_module = importlib.import_module("08_ground_truth")
evaluation_module = importlib.import_module("09_evaluation")

# قراءة مفاتيح الإعداد من Streamlit Secrets عند النشر
try:
    if not prompting.OPENROUTER_API_KEY:
        prompting.OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    prompting.OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", prompting.OPENROUTER_MODEL)
except Exception:
    pass

st.set_page_config(page_title="أسوان نوباوي  RAG", page_icon="⛵🐪🌊", layout="centered")
st.markdown(
    "<style>body, .stApp { direction: rtl; text-align: right; }</style>",
    unsafe_allow_html=True,
)

st.title("🏛️ أسوان RAG")
st.caption("نظام استرجاع وتوليد معزز بالمعرفة (RAG) عن ثقافة ومعالم أسوان")

# تبويبات: شات + تقييم
tab_chat, tab_eval = st.tabs(["💬 الشات", "📊 التقييم (Ground Truth)"])

# ================== تبويب الشات ==================
with tab_chat:
    with st.expander("ℹ️ عن المشروع"):
        st.write(
            f"قاعدة المعرفة تحتوي على {len(documents_module.get_documents())} مستنداً أصلياً عن أسوان. "
            "كل إجابة تُبنى فقط من المستندات المسترجعة فعلياً بعد تمريرها على Context Filter "
            "وتذكر مصادرها في نهايتها."
        )

    # إعدادات الفلترة في الشريط الجانبي
    with st.sidebar:
        st.header("⚙️ إعدادات الاسترجاع")
        k_value = st.slider("عدد المستندات المسترجعة (k)", 1, 6, 3)
        max_dist = st.slider(
            "أقصى مسافة مسموح بها (Context Filter)", 0.1, 1.5, 0.8, 0.05
        )
        dedup = st.checkbox("إزالة التكرار على مستوى المستند", value=True)

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
                context = retrieve_module.retrieve_context(
                    query,
                    k=k_value,
                    persist=False,
                    max_distance=max_dist,
                    deduplicate_by_document=dedup,
                )
                if not context:
                    answer = (
                        "⚠️ مفيش مستندات قريبة كفاية من السؤال بعد تطبيق Context Filter. "
                        "جرب تخفف قيمة max_distance أو تعيد صياغة السؤال."
                    )
                else:
                    answer = prompting.generate_answer(query, context)
            st.markdown(answer)
            if context:
                with st.expander("📄 الـ Chunks المسترجعة (بعد الفلترة)"):
                    for c in context:
                        st.write(
                            f"[مصدر {c['document_id']}] ({c['category']}) "
                            f"distance={c['distance']:.3f}"
                        )
                        st.write(c["text"])
                        st.divider()

        st.session_state.messages.append({"role": "assistant", "content": answer})

# ================== تبويب التقييم ==================
with tab_eval:
    st.subheader("📊 تقييم النظام على Ground Truth")
    gt = ground_truth_module.get_ground_truth()
    st.write(
        f"مجموعة التقييم تحتوي على **{len(gt)}** سؤال مرجعي، لكل سؤال المستندات المتوقع "
        "استرجاعها والكلمات المفتاحية المتوقع ظهورها في الإجابة."
    )

    col1, col2 = st.columns(2)
    with col1:
        k_eval = st.number_input("k للتقييم", min_value=1, max_value=10, value=3)
    with col2:
        max_dist_eval = st.number_input(
            "max_distance للفلترة", min_value=0.1, max_value=2.0, value=0.8, step=0.05
        )

    if st.button("🚀 شغل التقييم (بدون فلترة vs مع فلترة)"):
        with st.spinner("جاري تشغيل التقييم على كل الأسئلة..."):
            raw_summary, _ = evaluation_module.evaluate_retrieval(
                k=k_eval, use_filter=False
            )
            filtered_summary, filtered_results = evaluation_module.evaluate_retrieval(
                k=k_eval, use_filter=True, max_distance=max_dist_eval
            )

        st.success("✅ التقييم اكتمل")

        st.markdown("### 📈 مقارنة النتائج")
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**بدون Context Filter**")
            st.metric("Precision@k", raw_summary["avg_precision@k"])
            st.metric("Recall@k", raw_summary["avg_recall@k"])
            st.metric("F1@k", raw_summary["avg_f1@k"])
            st.metric("MRR", raw_summary["mrr"])
            st.metric("Hit Rate", raw_summary["hit_rate"])
        with col_b:
            st.markdown("**مع Context Filter**")
            st.metric("Precision@k", filtered_summary["avg_precision@k"])
            st.metric("Recall@k", filtered_summary["avg_recall@k"])
            st.metric("F1@k", filtered_summary["avg_f1@k"])
            st.metric("MRR", filtered_summary["mrr"])
            st.metric("Hit Rate", filtered_summary["hit_rate"])

        st.markdown("### 📋 تفاصيل كل سؤال (مع الفلترة)")
        for r in filtered_results:
            status = "✅" if r["reciprocal_rank"] > 0 else "❌"
            with st.expander(f"{status} {r['question']}"):
                st.write(f"**المتوقع:** {r['expected']}")
                st.write(f"**المسترجع:** {r['retrieved']}")
                st.write(
                    f"**Precision:** {r['precision']:.2f} | "
                    f"**Recall:** {r['recall']:.2f} | "
                    f"**F1:** {r['f1']:.2f}"
                )

    with st.expander("👀 استعرض أسئلة Ground Truth"):
        for item in gt:
            st.write(
                f"**{item['id']}.** {item['question']} → "
                f"docs: {item['expected_document_ids']} | "
                f"category: {item['category']}"
            )
