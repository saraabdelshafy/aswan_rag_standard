"""
08_ground_truth.py
المرحلة الثامنة: مجموعة بيانات مرجعية (Ground Truth) لتقييم أداء نظام RAG.

كل عنصر في المجموعة يحتوي على:
- question: سؤال المستخدم.
- expected_document_ids: قائمة IDs المستندات الأصلية اللي المفروض الاسترجاع يجيبها (relevant docs).
- expected_answer_keywords: كلمات مفتاحية المفروض تظهر في الإجابة النهائية للتحقق من صحتها.
- category: الفئة الرئيسية للسؤال (لتحليل الأداء لكل فئة).

تُستخدم هذه المجموعة في:
- 09_evaluation.py لحساب Precision / Recall / F1 على مرحلة الاسترجاع.
- التحقق من جودة الإجابات المولّدة (Answer coverage).
"""

GROUND_TRUTH = [
    {
        "id": 1,
        "question": "فين يقع معبد فيلة ولمين كان مخصص؟",
        "expected_document_ids": [0, 1],
        "expected_answer_keywords": ["فيلة", "أجيليكيا", "إيزيس"],
        "category": "معابد",
    },
    {
        "id": 2,
        "question": "امتى بتحصل ظاهرة تعامد الشمس على معبد أبو سمبل؟",
        "expected_document_ids": [3],
        "expected_answer_keywords": ["فبراير", "أكتوبر", "رمسيس"],
        "category": "معابد",
    },
    {
        "id": 3,
        "question": "مين اللي بنى معبد أبو سمبل؟",
        "expected_document_ids": [2, 3],
        "expected_answer_keywords": ["رمسيس", "نفرتاري"],
        "category": "معابد",
    },
    {
        "id": 4,
        "question": "ايه هو السد العالي وليه اتبنى؟",
        "expected_document_ids": [9],
        "expected_answer_keywords": ["الفيضانات", "الكهرباء", "ناصر"],
        "category": "سدود",
    },
    {
        "id": 5,
        "question": "ايه هي جزيرة كتشنر؟",
        "expected_document_ids": [11, 12],
        "expected_answer_keywords": ["نباتية", "كتشنر", "نباتات"],
        "category": "حدائق ومتنزهات",
    },
    {
        "id": 6,
        "question": "ايه هي رحلة الفلوكة في أسوان؟",
        "expected_document_ids": [14, 15],
        "expected_answer_keywords": ["فلوكة", "شراعية", "الغروب"],
        "category": "نزهات نيلية",
    },
    {
        "id": 7,
        "question": "ايه هو المتحف النوبي؟",
        "expected_document_ids": [16],
        "expected_answer_keywords": ["النوبة", "تاريخ", "أثرية"],
        "category": "متاحف",
    },
    {
        "id": 8,
        "question": "ازاي بيرحب النوبيون بضيوفهم؟",
        "expected_document_ids": [19],
        "expected_answer_keywords": ["القهوة", "الشاي", "الضيافة"],
        "category": "ثقافة وعادات",
    },
    {
        "id": 9,
        "question": "ايه هو السبوع عند النوبيين؟",
        "expected_document_ids": [21],
        "expected_answer_keywords": ["السابع", "الطفل", "أغاني"],
        "category": "ثقافة وعادات",
    },
    {
        "id": 10,
        "question": "ايه هي طقوس ليلة الحناء النوبية؟",
        "expected_document_ids": [22, 43],
        "expected_answer_keywords": ["الحناء", "العروس", "كوفري"],
        "category": "طقوس الزواج",
    },
    {
        "id": 11,
        "question": "ايه هو المشروب الأشهر في أسوان؟",
        "expected_document_ids": [26],
        "expected_answer_keywords": ["الكركديه"],
        "category": "مطبخ محلي",
    },
    {
        "id": 12,
        "question": "ايه أشهر أطباق المطبخ النوبي؟",
        "expected_document_ids": [27],
        "expected_answer_keywords": ["الويكة", "الفتة", "التمر"],
        "category": "مطبخ محلي",
    },
    {
        "id": 13,
        "question": "ازاي بيتبني البيت النوبي التقليدي؟",
        "expected_document_ids": [34, 35],
        "expected_answer_keywords": ["الطين", "اللبن", "حوش"],
        "category": "عمارة نوبية",
    },
    {
        "id": 14,
        "question": "ايه دلالة الزخارف اللي بيرسمها النوبيون على بيوتهم؟",
        "expected_document_ids": [36, 37],
        "expected_answer_keywords": ["الحسد", "الحماية", "المثلث"],
        "category": "زخارف ورموز",
    },
    {
        "id": 15,
        "question": "ايه هو زي المرأة النوبية التقليدي؟",
        "expected_document_ids": [39],
        "expected_answer_keywords": ["التوب", "الشبارة", "جلباب"],
        "category": "أزياء تقليدية",
    },
    {
        "id": 16,
        "question": "ايه هو زي العريس النوبي يوم الزفاف؟",
        "expected_document_ids": [40],
        "expected_answer_keywords": ["أبيض", "عمامة", "المركوب"],
        "category": "أزياء تقليدية",
    },
    {
        "id": 17,
        "question": "منين جاء اسم النوبة؟",
        "expected_document_ids": [48],
        "expected_answer_keywords": ["نب", "الذهب", "مناجم"],
        "category": "خلفية تاريخية",
    },
    {
        "id": 18,
        "question": "ايه هي المجموعات الرئيسية لسكان النوبة؟",
        "expected_document_ids": [49],
        "expected_answer_keywords": ["الكنوز", "الفديجا", "العرب"],
        "category": "خلفية تاريخية",
    },
    {
        "id": 19,
        "question": "ايه هي المسلة الناقصة وفين موجودة؟",
        "expected_document_ids": [6, 7],
        "expected_answer_keywords": ["الجرانيت", "المحاجر", "مسلة"],
        "category": "معالم أثرية",
    },
    {
        "id": 20,
        "question": "امتى أحسن وقت للسياحة في أسوان؟",
        "expected_document_ids": [29],
        "expected_answer_keywords": ["الشتوية", "دافئ", "جاف"],
        "category": "طبيعة",
    },
]


def get_ground_truth():
    """يرجع مجموعة البيانات المرجعية للتقييم."""
    return GROUND_TRUTH


if __name__ == "__main__":
    gt = get_ground_truth()
    print(f"عدد أسئلة التقييم (Ground Truth): {len(gt)}")
    print(f"الفئات المغطاة: {len(set(item['category'] for item in gt))}")
    print("\nمثال على عنصر:")
    print(gt[0])
