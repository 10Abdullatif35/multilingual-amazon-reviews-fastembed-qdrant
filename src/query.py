# src/query.py  (örnek kullanım)
# Qdrant üzerinde örnek bir vektör arama işlemi gösterir.

from fastembed import TextEmbedding
from src.qdrant_setup import client          # aynı client'i kullanıyoruz
from src.config import settings

# 1) Sorgu vektörünü üret
embedder = TextEmbedding(settings.MODEL_NAME, device=settings.DEVICE)  # Embedding modeli başlatılır

query_text = "Excellent quality and stellar service—highly recommend!"
#  Sorgulanacak metin (örnek)
query_vec  = embedder.embed([query_text])               #  Metni embed ederek vektörünü üret

# 2) Yalnızca İngilizce shard'ında (en) ara
"""
hits = client.search(
    collection_name=settings.COLLECTION,
    query_vector=query_vec,           
    shard_key_selector="es",   
    with_payload=True,
)  
DeprecationWarning: `search` method is deprecated and will be removed in the future. Use `query_points` instead.                    
"""
# Modern ve önerilen yöntemle sorgu (query_points)
hits = client.query_points(
    collection_name=settings.COLLECTION,      # Hangi koleksiyonda arama yapılacak
    query=query_vec,           # Sorgu vektörü (embedding)
    shard_key_selector="en",   # Sadece İngilizce shard'ında ara
    limit=5,                   # En fazla 5 sonuç getir
    with_payload=True,         # Sonuçlarda ek veri (payload) da getir
).points                       # Sonuçları .points ile alın


# 3) Sonuçları yazdır
for h in hits:
    print(
        f"[{h.payload['language']}] ★{h.payload['stars']}  score={h.score:.3f}"
    )

# ÖR. çıktı:
# [en] ★5  score=0.812
# [en] ★4  score=0.799
