"""
07_prompting.py
المرحلة السابعة: بناء الـ Prompt من السياق المسترجع، واستدعاء نموذج لغوي
عبر OpenRouter API لتوليد إجابة نهائية، مع إلحاق المصادر المستخدمة إجبارياً
بنهاية كل إجابة (لضمان أن الإجابة تستشهد بمصادرها دائماً).

"""
import os
import re

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

FALLBACK_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openrouter/free",
]


def build_prompt(query, context):
    """يبني الـ Prompt النهائي مع ترقيم كل مصدر ليتمكن النموذج من الاستشهاد به."""
    context_text = "\n".join(
        f"[مصدر {c['document_id']}] ({c['category']}) {c['text']}" for c in context
    )
    return f"""أنت مرشد سياحي وثقافي متخصص في مدينة أسوان، تتحدث بأسلوب ودود ومرحّب بالعربية، وتستخدم رموزاً تعبيرية (إيموجي) مناسبة تضفي دفئاً على الإجابة.

قواعد يجب الالتزام بها:
- ابدأ الإجابة بترحيب قصير ودافئ بالزائر مع رمز تعبيري مناسب، ثم انتقل للمعلومة.
- اكتب الإجابة بالكامل باللغة العربية الفصحى، ولا تُدرج أي كلمة أو حرف باللغة الإنجليزية أو أي لغة أخرى داخل النص إطلاقاً، إلا إذا كانت مذكورة حرفياً بنفس الصيغة في نص السياق أدناه.
- اذكر رقم المصدر بين قوسين مربعين بعد كل معلومة تستخدمها، مثال: [مصدر 3].
- استخدم فقط المصادر التي استشهدت بها فعلياً داخل نص الإجابة؛ لا تذكر أي مصدر من السياق أدناه لم تستخدمه.
- إذا كانت المعلومات المتوفرة في السياق غير متعلقة بالسؤال أو غير كافية للإجابة عليه، صرّح بذلك بوضوح ومباشرة دون اختلاق معلومات، ودون استخدام أي من المصادر غير ذات الصلة.

السياق المسترجع:
{context_text}

سؤال المستخدم: {query}

الإجابة:"""


def _call_openrouter_once(prompt, model):
    """يستدعي OpenRouter بموديل واحد محدد، ويرجع (نجاح: bool, نص: str)."""
    response = requests.post(
        OPENROUTER_URL,
        headers={
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=60,
    )
    if response.status_code == 404:
        return False, "model_not_found"
    response.raise_for_status()
    data = response.json()
    return True, data["choices"][0]["message"]["content"].strip()


def call_openrouter(prompt):
    """
    يستدعي OpenRouter API بالموديل المضبوط في OPENROUTER_MODEL.
    لو الموديل غير موجود في الكتالوج (404)، يجرب الموديلات في FALLBACK_MODELS بالترتيب تلقائياً
    قبل إرجاع رسالة خطأ نهائية.
    """
    if not OPENROUTER_API_KEY:
        return "⚠️ لم يتم ضبط مفتاح OPENROUTER_API_KEY بعد."

    models_to_try = [OPENROUTER_MODEL] + [m for m in FALLBACK_MODELS if m != OPENROUTER_MODEL]

    last_error = None
    for model in models_to_try:
        try:
            ok, result = _call_openrouter_once(prompt, model)
            if ok:
                return result
            last_error = f"الموديل {model} غير متاح حالياً (404)"
        except Exception as e:
            last_error = str(e)

    return f"❌ خطأ أثناء الاتصال بـ OpenRouter بعد تجربة كل الموديلات المتاحة: {last_error}"


def _extract_cited_ids(answer):
    """يستخرج أرقام المصادر [مصدر N] اللي فعلاً ظهرت داخل نص الإجابة."""
    return set(re.findall(r"\[مصدر\s+(\d+)\]", answer))


def format_sources(context, answer=""):
    """
    يبني قائمة نصية بالمصادر التي استُشهد بها فعلياً داخل نص الإجابة فقط.
    لو answer فاضية أو مفيهاش أي [مصدر N]، بيرجع لعرض كل الـ context (توافقاً
    مع الاستخدام القديم)، لكن الحالة الطبيعية هي فلترة المصادر غير المستخدمة.
    """
    cited_ids = _extract_cited_ids(answer)
    relevant = [c for c in context if not cited_ids or str(c["document_id"]) in cited_ids]
    lines = [f"- مصدر {c['document_id']} ({c['category']})" for c in relevant]
    return "**المصادر المستخدمة:**\n" + "\n".join(lines)


REFUSAL_KEYWORDS = [
    "لا أملك معلومات",
    "لا أعرف",
    "ليس لدي أي معلومات",
    "لا تتوفر",
    "لا يوجد في السياق",
    "لم يتم ذكر",
    "أعتذر",
    "اعتذر",
    "عذرًا",
    "غير موجود في السياق",
    "لا يمكنني الإجابة",
]


def is_refusal(answer):
    """
    يتحقق هل الإجابة تمثّل رفضاً (عدم توفر معلومات كافية) أم إجابة فعلية.
    يُستخدم خارج هذا الملف (مثلاً في streamlit_app.py) لتقرير هل نعرض
    صندوق "المصادر المسترجعة" للمستخدم أم لا — لأنه حتى لو تم استرجاع
    chunks (context غير فارغ)، ممكن تكون كلها غير كافية للإجابة فعلياً،
    فيرفض النموذج، وحينها المصادر دي متبقاش مصادر "استُخدمت" فعلاً.
    """
    if not answer:
        return False
    return any(keyword in answer for keyword in REFUSAL_KEYWORDS)


def _strip_stray_english(text):
    """
    طبقة حماية إضافية: تشيل أي كلمة إنجليزية منفردة اتقحمت وسط جملة عربية
    (بين كلمتين عربيتين)، كنتيجة هلوسة من النموذج، مع الحفاظ على أي مصطلح
    إنجليزي أصلي مذكور كجزء من جملة إنجليزية أطول (أكتر من كلمة متتالية).
    """
    arabic = "\u0600-\u06FF"
    pattern = re.compile(rf"(?<=[{arabic}])(\s+)[A-Za-z]{{2,15}}(?!\s*[A-Za-z])")
    cleaned = pattern.sub(lambda m: m.group(1), text)
    return re.sub(r"[ \t]{2,}", " ", cleaned)


def generate_answer(query, context):
    """يستدعي النموذج، ويرجع الإجابة النهائية مع قائمة المصادر إن وجدت."""

    prompt = build_prompt(query, context)
    answer = call_openrouter(prompt)
    answer = _strip_stray_english(answer)

    if not context or is_refusal(answer):
        return answer

    return f"{answer}\n\n{format_sources(context, answer)}"


if __name__ == "__main__":
    import importlib

    retrieve_module = importlib.import_module("06_retrieve_context")
    query = "عادات وتقاليد أهل النوبة في الاحتفالات"
    context = retrieve_module.retrieve_context(query, k=3)
    print(generate_answer(query, context))
