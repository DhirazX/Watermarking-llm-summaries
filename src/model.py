# Import summarization model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

def get_model():
    # bart is already fine tuned for summarization
    model_name = "facebook/bart-large-cnn"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

    return tokenizer, model, device


def summarize(article, tokenizer, model, device):
    inputs = tokenizer(article, max_length=1024, truncation=True, return_tensors="pt").to(device)

    output = model.generate(**inputs, max_length=128, num_beams=4)

    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary


if __name__ == "__main__":
    tokenizer, model, device = get_model()

    test_article = "Scientists announced on Tuesday that they had discovered a new species of frog in the Amazon rainforest. The frog glows under ultraviolet light and has two heads with four eyes and is believed to be related to other fluorescent amphibians in the region especially the neon green-headed salamander."

    print("device:", device)
    print("summary:", summarize(test_article, tokenizer, model, device))