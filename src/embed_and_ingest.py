"""
Yerel Parquet dosyalarından okuyup embedding + Qdrant upsert yapan script.
"""
import os
from uuid import uuid4
import pyarrow.parquet as pq
from fastembed import TextEmbedding
from qdrant_client import models
from loguru import logger

from src.config import settings
from src.qdrant_setup import client, init_collection

# Veri dosyalarının bulunduğu klasör (proje kökünde 'data')
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
#LANGS      = ["en", "de", "fr", "es", "ja", "zh"]
LANGS      = ["fr", "es", "ja", "zh"]  # Yüklenecek dillerin listesi (örnek olarak bir kısmı aktif)
BATCH_SIZE = 1024  # Her seferde işlenecek satır sayısı (batch)

def iter_parquet_rows(path, batch_size):
    """
    Parquet dosyasını batch'ler halinde okur ve her batch'te metin ve yıldız puanlarını döneryor.
    Metin ve puan kolonlarının isimleri farklı olabileceği için esnek kontrol yapar.
    """
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        d = batch.to_pydict()

        # --- METİN ---
        if "review_body" in d:
            texts = d["review_body"]
        elif "text" in d:
            texts = d["text"]
        else:
            raise KeyError("Metin kolonu bulunamadı ('review_body' veya 'text')")

        # --- YILDIZ / LABEL ---
        if "stars" in d:
            stars = d["stars"]
        elif "label" in d:           # 0-4 → 1-5’e çevir
            stars = [int(x) + 1 for x in d["label"]]
        else:
            raise KeyError("Puan kolonu bulunamadı ('stars' veya 'label')")

        yield texts, stars


def main():
    # Qdrant koleksiyonunu ve shard'ları başlat
    init_collection()
    # Embedding modeli başlatılır
    embedder = TextEmbedding(settings.MODEL_NAME, device=settings.DEVICE)

    for lang in LANGS:
        # Her dil için ilgili Parquet dosyasının yolunu oluştur
        parquet_path = os.path.join(DATA_DIR, f"{lang}.parquet")
        if not os.path.exists(parquet_path):
            logger.error(f"{parquet_path} bulunamadı; önce download_data.py çalıştırman gerek.")
            continue

        logger.info(f"➡️  {lang} shard'ına yükleniyor…")
        total = 0

        # Parquet dosyasını batch'ler halinde oku ve Qdrant'a yükle
        for texts, stars in iter_parquet_rows(parquet_path, BATCH_SIZE):
            # Her metin için embedding vektörü üret
            vecs = list(embedder.embed(texts))
            # Her embedding ve puan için Qdrant PointStruct nesnesi oluştur
            points = [
                models.PointStruct(
                    id=str(uuid4()),
                    vector=v,
                    payload={"language": lang, "stars": int(s)}
                )
                for v, s in zip(vecs, stars)
            ]
            # Qdrant'a batch olarak upsert işlemi (shard-key: dil)
            client.upsert(
                collection_name=settings.COLLECTION,
                points=points,
                shard_key_selector=lang,   # Dil = shard-key
            )
            total += len(points)

        logger.success(f"{lang}: {total:,} kayıt yüklendi.")

if __name__ == "__main__":
    main()

