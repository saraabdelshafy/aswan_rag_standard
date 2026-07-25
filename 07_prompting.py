"""
07_prompting.py
المرحلة السابعة: بناء الـ Prompt من السياق المسترجع، واستدعاء نموذج لغوي
عبر OpenRouter API لتوليد إجابة نهائية، مع إلحاق المصادر المستخدمة إجبارياً
بنهاية كل إجابة (لضمان أن الإجابة تستشهد بمصادرها دائماً).

لا يوجد أي مفتاح API مكتوب هنا صراحة — القيم تُقرأ من متغيرات البيئة، وعند
النشر على Streamlit Cloud تُقرأ من Streamlit Secrets (انظر streamlit_app.py).
"""
import os

import requests

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
# نموذج مجاني افتراضياً (لاحقة :free) لتفادي أي تكلفة أثناء التجربة والتسليم
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# قائمة احتياطية: لو الموديل الأساسي اتشال من قائمة OpenRouter المجانية (كتالوج الموديلات المجانية
# يتغيّر باستمرار)، الكود يجرب الموديلات دي بالترتيب تلقائياً قبل ما يستسلم.
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
    return f"""أنت مرشد سياحي وثقافي متخصص في مدينة أسوان، تتحدث بأسلوب واضح وودود بالعربية.
أجب عن سؤال المستخدم بالاعتماد فقط على المعلومات الموجودة في السياق أدناه.
اذكر رقم المصدر بين قوسين مربعين بعد كل معلومة تستخدمها، مثال: [مصدر 3].
إذا لم تكفِ المعلومات المتوفرة للإجابة، صرّح بذلك بوضوح ولا تختلق معلومات من عندك.

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


def format_sources(context):
    """يبني قائمة نصية بالمصادر المستخدمة، تُلحق دائماً بنهاية الإجابة."""
    lines = [f"- مصدر {c['document_id']} ({c['category']})" for c in context]
    return "**المصادر المستخدمة:**\n" + "\n".join(lines)


def generate_answer(query, context):
    """يبني الـ Prompt، يستدعي النموذج، ويرجع الإجابة النهائية مع قائمة المصادر دائماً."""
    prompt = build_prompt(query, context)
    answer = call_openrouter(prompt)
    return f"{answer}\n\n{format_sources(context)}"


if __name__ == "__main__":
    import importlib

    retrieve_module = importlib.import_module("06_retrieve_context")
    query = "عادات وتقاليد أهل النوبة في الاحتفالات"
    context = retrieve_module.retrieve_context(query, k=3)
    print(generate_answer(query, context))
