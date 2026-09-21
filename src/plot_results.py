# plot summary length vs zscore
import json
import matplotlib.pyplot as plt

with open("results.json") as f:
    results = json.load(f)

lengths = [r["wm_token_count"] for r in results]
zscores = [r["wm_zscore"] for r in results]

plt.scatter(lengths, zscores)
plt.xlabel("summary length (tokens)")
plt.ylabel("z score")
plt.title("does summary length affect watermark detection")
plt.axhline(y=4, color="r", linestyle="--")  # just a rough line to eyeball whats "clearly detected"
plt.savefig("length_vs_zscore.png")
plt.show()

print("saved plot as length_vs_zscore.png")