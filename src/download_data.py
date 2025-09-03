import os
from datasets import load_dataset, Dataset

# Çıktı dosyalarının kaydedileceği klasör (proje kökünde 'data')
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)  # Klasör yoksa oluştur

# İndirilecek dillerin listesi
LANGS = ["en", "de", "fr", "es", "ja", "zh"]


def save(ds: Dataset, lang: str):
    """
    Verilen HuggingFace Dataset nesnesini Parquet formatında kaydeder.
    Her dil için ayrı dosya oluşturur.
    """
    path = os.path.join(OUT_DIR, f"{lang}.parquet")
    ds.to_parquet(path)
    print(f"  {lang:<2} → {path}  ({len(ds):,} satır)")

# 1  Öncelikle MTEB veri setinin aynasını indirmeyi dene
try:
    print("  mteb/amazon_reviews_multi deneniyor…")
    for lang in LANGS:
        ds_lang = load_dataset(
            "mteb/amazon_reviews_multi",
            name=lang,          # ← Dil konfigürasyonu (örn: 'en', 'de', ...)
            split="train"
        )
        save(ds_lang, lang)
    print("  MTEB aynasından indirme tamam.")
    exit(0)  # Başarılıysa programı bitir
except Exception as e:
    print(f"   MTEB aynası başarısız: {e}")

# 2  Eğer ilk kaynak başarısız olursa, yedek veri setini indir
print("  srvmishra832/multilingual-amazon-reviews-6-languages deneniyor…")
ds_all = load_dataset(
    "srvmishra832/multilingual-amazon-reviews-6-languages",
    split="train"
)
for lang in LANGS:
    # Her dil için filtre uygula ve ayrı dosya olarak kaydet
    save(
        ds_all.filter(lambda r: r["language"] == lang, keep_in_memory=False),
        lang
    )

print("  İndirme & Parquet kaydetme tamam.")
