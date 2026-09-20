# watermark logic
import torch

GAMMA = 0.25   # fraction of vocab thats green
DELTA = 2.0    # amount we boost green tokens by
SECRET_KEY = 15485863   # random key
VOCAB_SIZE = 50265      # bart's vocab size


def get_green_list(prev_token, vocab_size=VOCAB_SIZE, gamma=GAMMA, seed_key=SECRET_KEY):
    # seed a random generator using prev token + secret key
    seed = seed_key * prev_token.item()
    g = torch.Generator()
    g.manual_seed(seed)

    # shuffle all vocab ids, take first chunk as green list
    perm = torch.randperm(vocab_size, generator=g)
    green_size = int(vocab_size * gamma)
    green_ids = perm[:green_size]
    return green_ids


def apply_watermark(input_ids, scores, delta=DELTA):
    vocab_size = scores.shape[-1]  

    for i in range(input_ids.shape[0]):
        prev_token = input_ids[i][-1]
        green_ids = get_green_list(prev_token, vocab_size=vocab_size)
        green_ids = green_ids.to(scores.device)  # move to same device as scores (cuda or cpu)
        scores[i][green_ids] += delta

    return scores

# quick test
if __name__ == "__main__":
    fake_input_ids = torch.tensor([[10, 25]])
    fake_scores = torch.zeros((1, VOCAB_SIZE))

    new_scores = apply_watermark(fake_input_ids, fake_scores)
    print(new_scores[0][:20])