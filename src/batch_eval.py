# batch evaluation, running everything on a bunch of articles
from dataset import get_data
from model import get_model, summarize, summarize_watermarked
from detector import detect_watermark
from rouge_score import rouge_scorer

def run_batch(n=50):
    data = get_data(n)
    tokenizer, model, device = get_model()

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = []

    for i, d in enumerate(data):
        print("doing article", i, "/", n)

        article = d["article"]
        reference = d["reference_summary"]

        # plain summary
        plain_summary = summarize(article, tokenizer, model, device)
        plain_scores = scorer.score(reference, plain_summary)
        plain_detect = detect_watermark(plain_summary, tokenizer, model)

        # watermarked summary
        wm_summary = summarize_watermarked(article, tokenizer, model, device)
        wm_scores = scorer.score(reference, wm_summary)
        wm_detect = detect_watermark(wm_summary, tokenizer, model)

        results.append({
            "article": article,
            "reference": reference,

            "plain_summary": plain_summary,
            "plain_rouge1": plain_scores["rouge1"].fmeasure,
            "plain_zscore": plain_detect["z_score"],

            "wm_summary": wm_summary,
            "wm_rouge1": wm_scores["rouge1"].fmeasure,
            "wm_zscore": wm_detect["z_score"],
            "wm_token_count": wm_detect["total_tokens"],  # track length so we can check short vs long 
        })

    return results


if __name__ == "__main__":
    results = run_batch(150)

    avg_plain_rouge1 = sum(r["plain_rouge1"] for r in results) / len(results)
    avg_wm_rouge1 = sum(r["wm_rouge1"] for r in results) / len(results)

    avg_plain_z = sum(r["plain_zscore"] for r in results) / len(results)
    avg_wm_z = sum(r["wm_zscore"] for r in results) / len(results)

    print()
    print("avg plain rouge1:", avg_plain_rouge1)
    print("avg watermarked rouge1:", avg_wm_rouge1)
    print()
    print("avg plain zscore:", avg_plain_z)
    print("avg watermarked zscore:", avg_wm_z)

    # save results so we can plot length vs zscore later
    import json
    with open("results.json", "w") as f:
        json.dump(results, f, indent=2)

    print()
    print("saved results.json")