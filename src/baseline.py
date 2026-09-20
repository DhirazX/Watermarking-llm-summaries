# baseline run, no watermark yet
from dataset import get_data
from model import get_model, summarize
from rouge_score import rouge_scorer

def run_baseline(n=10):
    data = get_data(n)
    tokenizer, model, device = get_model()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = []
    for d in data:
        pred_summary = summarize(d["article"], tokenizer, model, device)
        scores = scorer.score(d["reference_summary"], pred_summary)

        results.append({
            "article": d["article"],
            "reference": d["reference_summary"],
            "prediction": pred_summary,
            "rouge1": scores["rouge1"].fmeasure,
            "rouge2": scores["rouge2"].fmeasure,
            "rougeL": scores["rougeL"].fmeasure,
        })

    return results


if __name__ == "__main__":
    results = run_baseline(5)

    for r in results:
        print("reference:", r["reference"])
        print("prediction:", r["prediction"])
        print("rouge1:", r["rouge1"], "rouge2:", r["rouge2"], "rougeL:", r["rougeL"])
        print()

    avg_r1 = sum(r["rouge1"] for r in results) / len(results)
    print("avg rouge1:", avg_r1)