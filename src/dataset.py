from datasets import load_dataset

def get_data(n=200):
    # loading cnn dailymail dataset test split
    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split="test")    
    ds = ds.shuffle(seed=42).select(range(n))

    data = []
    for row in ds:
        article = row["article"]
        summary = row["highlights"]
        data.append({"article": article, "reference_summary": summary})

    return data


if __name__ == "__main__":
    data = get_data(3)
    for d in data:
        print(d["article"][:200])
        print(d["reference_summary"])
        print()