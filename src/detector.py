# detector logic
import torch
from watermark import get_green_list, GAMMA

def detect_watermark(text, tokenizer, model, gamma=GAMMA):
    ids = tokenizer(text, return_tensors="pt")["input_ids"][0]

    green_count = 0
    total = 0

    vocab_size = model.config.vocab_size  # use model's real vocab size, not tokenizer's

    for i in range(1, len(ids)):
        prev_token = ids[i - 1]
        current_token = ids[i]

        green_ids = get_green_list(prev_token, vocab_size=vocab_size)

        if current_token in green_ids:
            green_count += 1

        total += 1

    expected = total * gamma
    std = (total * gamma * (1 - gamma)) ** 0.5
    z_score = (green_count - expected) / std

    return {
        "green_count": green_count,
        "total_tokens": total,
        "green_fraction": green_count / total,
        "z_score": z_score,
    }

if __name__ == "__main__":
    from model import get_model, summarize, summarize_watermarked

    tokenizer, model, device = get_model()

    test_article = "Scientists announced on Tuesday that they had discovered a new species of frog in the Amazon rainforest. The frog glows under ultraviolet light and has two heads with four eyes and is believed to be related to other fluorescent amphibians in the region especially the neon green-headed salamander."

    plain_summary = summarize(test_article, tokenizer, model, device)
    wm_summary = summarize_watermarked(test_article, tokenizer, model, device)

    print("plain detection:", detect_watermark(plain_summary, tokenizer, model))
    print("watermarked detection:", detect_watermark(wm_summary, tokenizer, model))

    print("tokenizer vocab_size:", tokenizer.vocab_size)
    print("model real vocab_size:", model.config.vocab_size)