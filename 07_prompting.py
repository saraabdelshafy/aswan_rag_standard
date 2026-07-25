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
OPENROUTER_MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct:free")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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


def call_openrouter(prompt):
    """يستدعي OpenRouter API ويرجع نص الإجابة، أو رسالة خطأ واضحة لو فشل الاتصال."""
    if not OPENROUTER_API_KEY:
        return "⚠️ لم يتم ضبط مفتاح OPENROUTER_API_KEY بعد."
    try:
        response = requests.post(
            OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": OPENROUTER_MODEL,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"❌ خطأ أثناء الاتصال بـ OpenRouter: {e}"


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
