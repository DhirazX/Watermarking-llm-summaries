# watermarking-llm-summaries

An attempt to recreate the LLM watermarking algorithm from this paper (https://arxiv.org/abs/2301.10226) and apply it to a pretrained summarization model (BART), to see how well it holds up on short, summarized text.

## What this does

The watermark works by biasing the model towards a random subset of "green" words at each generation step, based on a hash of the previous word. A separate detector can then check any piece of text and tell (without needing the model) whether it was watermarked or not, using basic statistics (z-score).

## What I found

- Watermarking barely hurt summary quality. ROUGE-1 dropped only slightly (0.432 to 0.422) compared to normal summaries.
- Watermarked summaries were detected reliably. Average z-score jumped from about 0 (normal text) to about 4.1 (watermarked text), which means its very unlikely to be random chance.
- Tried to see if shorter summaries are harder to detect than longer ones. Within the length range this model naturally produces (50-125 tokens), detection strength stayed fairly flat, no clear pattern either way. Would probably need a wider range of lengths to properly test this.

## Setup

```bash
pip install -r requirements.txt
```
