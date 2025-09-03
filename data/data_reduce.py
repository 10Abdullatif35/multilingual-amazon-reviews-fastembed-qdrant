# data_reduce.py  ─ çalıştırma dizini:  data\
import os, glob, math, json
import pyarrow.parquet as pq
from collections import Counter
from datasets import Dataset, concatenate_datasets

DATA_DIR = os.path.dirname(__file__)

TARGET = {
    "fr": 70_000,    # mutlak adet
    "es": 70_000,
    "ja": 30_000,
    "zh": 30_000,
    # en / de belirtilmedi ⇒ atlanır
}

def stratified_sample(ds: Dataset, n: int | float) -> Dataset:
    rating_col = "stars" if "stars" in ds.column_names else "label"

    # ORAN verildiyse (float)
    if isinstance(n, float):
        return ds.train_test_split(
            test_size=n,
            stratify_by_column=rating_col,
            seed=42,
        )["test"]

    # MUTLAK adet (int)
    counts = Counter(ds[rating_col])  # {1: 45000, 2: ...}
    per_star = {s: math.floor(c * n / len(ds)) for s, c in counts.items()}

    buckets = []
    for star, k in per_star.items():
        subset = ds.filter(lambda r: r[rating_col] == star, keep_in_memory=False)
        buckets.append(subset.shuffle(seed=42).select(range(k)))

    sample = concatenate_datasets(buckets)

    # Eksiğimiz varsa rastgele tamamla
    missing = n - len(sample)
    if missing > 0:
        extra = ds.filter(lambda r: r[rating_col] not in sample[rating_col], keep_in_memory=False)
        sample = concatenate_datasets(
            [sample, extra.shuffle(seed=42).select(range(missing))]
        )

    # label kolonunu stars'a dönüştür
    if rating_col == "label":
        sample = sample.map(
            lambda r: {"stars": int(r["label"]) + 1},
            remove_columns=["label"],
            num_proc=1,
        )
    return sample

for path in glob.glob(os.path.join(DATA_DIR, "*.parquet")):
    lang = os.path.basename(path).split(".")[0]
    rule = TARGET.get(lang)

    if rule is None:
        print(f"{lang}: atlanıyor (değiştirilmedi)")
        continue

    print(f"{lang}: örnekleniyor…")
    ds = Dataset(pq.read_table(path))
    sampled = stratified_sample(ds, rule)

    out_path = path.replace(".parquet", "_sample.parquet")
    sampled.to_parquet(out_path)
    print(f"{lang}: {len(sampled):,} satır → {out_path}")

print("✅  Tüm örnekleme tamamlandı.")
