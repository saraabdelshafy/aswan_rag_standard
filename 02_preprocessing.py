"""
02_preprocessing.py
المرحلة الثانية: تنظيف وتوحيد شكل النصوص العربية قبل التقطيع والتمثيل المتجهي.
"""
import importlib
import re

documents_module = importlib.import_module("01_documents")


def preprocess_arabic(text):
    """توحيد شكل الكلمات العربية (إزالة تشكيل، توحيد الألف/الياء/التاء المربوطة، إزالة ترقيم)."""
    text = re.sub(r"[\u064B-\u065F]", "", text)   # إزالة التشكيل
    text = re.sub(r"[إأآا]", "ا", text)            # توحيد أشكال الألف
    text = re.sub(r"ى", "ي", text)                 # توحيد الألف المقصورة بالياء
    text = re.sub(r"ة", "ه", text)                 # توحيد التاء المربوطة بالهاء
    text = re.sub(r"[^\w\s]", "", text)            # إزالة علامات الترقيم
    text = re.sub(r"\s+", " ", text).strip()       # إزالة المسافات الزائدة
    return text


def preprocess_documents(documents=None):
    """يرجع نسخة من المستندات مع نص منظّف إضافي في المفتاح clean_text."""
    documents = documents if documents is not None else documents_module.get_documents()
    return [{**doc, "clean_text": preprocess_arabic(doc["text"])} for doc in documents]


if __name__ == "__main__":
    cleaned_docs = preprocess_documents()
    print("قبل التنظيف:", cleaned_docs[0]["text"])
    print("بعد التنظيف :", cleaned_docs[0]["clean_text"])
