"""
09_evaluation.py
المرحلة التاسعة: تقييم أداء نظام RAG باستخدام Ground Truth.

يقيس هذا الملف:
1. أداء الاسترجاع (Retrieval Metrics):
   - Precision@k: نسبة المستندات المسترجعة اللي فعلاً relevant.
   - Recall@k: نسبة المستندات الـ relevant اللي النظام قدر يجيبها.
   - F1@k: المتوسط التوافقي بين Precision و Recall.
   - Hit Rate: نسبة الأسئلة اللي جاب فيها النظام على الأقل مستند واحد صحيح.
   - MRR (Mean Reciprocal Rank): متوسط رتبة أول نتيجة صحيحة.

2. أداء الإجابة (Answer Metrics):
   - Keyword Coverage: نسبة الكلمات المفتاحية المتوقعة اللي ظهرت في الإجابة.

3. مقارنة بين الاسترجاع بدون فلترة (Raw) والاسترجاع مع Context Filter.
"""
import importlib

retrieve_module = importlib.import_module("06_retrieve_context")
ground_truth_module = importlib.import_module("08_ground_truth")


def precision_at_k(retrieved_doc_ids, expected_doc_ids):
    """نسبة المستندات المسترجعة اللي فعلاً relevant."""
    if not retrieved_doc_ids:
        return 0.0
    hits = sum(1 for d in retrieved_doc_ids if d in expected_doc_ids)
    return hits / len(retrieved_doc_ids)


def recall_at_k(retrieved_doc_ids, expected_doc_ids):
    """نسبة المستندات الـ relevant اللي النظام قدر يجيبها."""
    if not expected_doc_ids:
        return 0.0
    hits = sum(1 for d in expected_doc_ids if d in retrieved_doc_ids)
    return hits / len(expected_doc_ids)


def f1_score(precision, recall):
    """المتوسط التوافقي بين Precision و Recall."""
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def reciprocal_rank(retrieved_doc_ids, expected_doc_ids):
    """رتبة أول نتيجة صحيحة (1/rank)، أو 0 لو مفيش."""
    for i, d in enumerate(retrieved_doc_ids, start=1):
        if d in expected_doc_ids:
            return 1.0 / i
    return 0.0


def keyword_coverage(answer_text, expected_keywords):
    """نسبة الكلمات المفتاحية المتوقعة اللي ظهرت في الإجابة."""
    if not expected_keywords:
        return 0.0
    answer_lower = answer_text.lower()
    hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return hits / len(expected_keywords)


def evaluate_retrieval(k=3, use_filter=True, max_distance=0.8):
    """
    يقيّم أداء مرحلة الاسترجاع على كل أسئلة الـ Ground Truth.

    use_filter=True  -> يستخدم Context Filter (max_distance + dedup).
    use_filter=False -> استرجاع خام بدون فلترة (للمقارنة).
    """
    gt = ground_truth_module.get_ground_truth()
    results = []

    for item in gt:
        if use_filter:
            context = retrieve_module.retrieve_context(
                item["question"],
                k=k,
                max_distance=max_distance,
                deduplicate_by_document=True,
            )
        else:
            context = retrieve_module.retrieve_context(
                item["question"], k=k, max_distance=999
            )

        retrieved_ids = [c["document_id"] for c in context]
        expected_ids = item["expected_document_ids"]

        p = precision_at_k(retrieved_ids, expected_ids)
        r = recall_at_k(retrieved_ids, expected_ids)
        f = f1_score(p, r)
        rr = reciprocal_rank(retrieved_ids, expected_ids)

        results.append(
            {
                "question": item["question"],
                "expected": expected_ids,
                "retrieved": retrieved_ids,
                "precision": p,
                "recall": r,
                "f1": f,
                "reciprocal_rank": rr,
            }
        )

    # حساب المتوسطات
    n = len(results)
    avg_precision = sum(r["precision"] for r in results) / n
    avg_recall = sum(r["recall"] for r in results) / n
    avg_f1 = sum(r["f1"] for r in results) / n
    mrr = sum(r["reciprocal_rank"] for r in results) / n
    hit_rate = sum(1 for r in results if r["reciprocal_rank"] > 0) / n

    summary = {
        "num_questions": n,
        "k": k,
        "use_filter": use_filter,
        "avg_precision@k": round(avg_precision, 3),
        "avg_recall@k": round(avg_recall, 3),
        "avg_f1@k": round(avg_f1, 3),
        "mrr": round(mrr, 3),
        "hit_rate": round(hit_rate, 3),
    }
    return summary, results


def evaluate_answers(k=3):
    """يقيّم جودة الإجابات المولّدة عبر Keyword Coverage (اختياري - يستدعي LLM)."""
    prompting = importlib.import_module("07_prompting")
    gt = ground_truth_module.get_ground_truth()
    scores = []

    for item in gt:
        context = retrieve_module.retrieve_context(
            item["question"], k=k, deduplicate_by_document=True
        )
        answer = prompting.generate_answer(item["question"], context)
        coverage = keyword_coverage(answer, item["expected_answer_keywords"])
        scores.append(
            {
                "question": item["question"],
                "keyword_coverage": coverage,
                "answer_preview": answer[:150],
            }
        )

    avg_coverage = sum(s["keyword_coverage"] for s in scores) / len(scores)
    return {"avg_keyword_coverage": round(avg_coverage, 3)}, scores


def print_summary(title, summary):
    print("=" * 60)
    print(title)
    print("=" * 60)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    # مقارنة: بدون فلترة vs مع فلترة
    print("\n📊 تقييم مرحلة الاسترجاع على Ground Truth\n")

    raw_summary, raw_results = evaluate_retrieval(k=3, use_filter=False)
    print_summary("❌ بدون Context Filter (Raw Retrieval)", raw_summary)

    filtered_summary, filtered_results = evaluate_retrieval(
        k=3, use_filter=True, max_distance=0.8
    )
    print_summary("✅ مع Context Filter (max_distance=0.8 + dedup)", filtered_summary)

    # تفاصيل لكل سؤال (مع الفلترة)
    print("=" * 60)
    print("تفاصيل الأداء لكل سؤال (مع الفلترة):")
    print("=" * 60)
    for r in filtered_results:
        status = "✅" if r["reciprocal_rank"] > 0 else "❌"
        print(
            f"{status} {r['question'][:50]}... "
            f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}"
        )
        print(f"   المتوقع: {r['expected']} | المسترجع: {r['retrieved']}")
