# src/qdrant_setup.py
# Qdrant istemcisi ve koleksiyon/shard anahtarı (shard-key) kurulumunu yöneten yardımcı dosya.

from qdrant_client import QdrantClient, models
from src.config import settings

# Qdrant veritabanına bağlantı kuran istemci (client) nesnesi
client = QdrantClient(
    url=str(settings.QDRANT_URL),           # Qdrant sunucu adresi
    api_key=settings.QDRANT_API_KEY,        # API anahtarı
    prefer_grpc=True,                       # gRPC protokolünü
)

def init_collection():
    """
    Qdrant'da koleksiyon yoksa oluşturur, varsa hiçbir şey yapmaz.
    Ayrıca her dil için shard-key ekler.
    """
    try:
        client.get_collection(settings.COLLECTION)
        # Koleksiyon zaten var → hiçbir şey yapma
        return
    except Exception:
        pass  # get_collection hata verdiyse oluştur

    # Koleksiyonu oluştur
    client.create_collection(
        collection_name=settings.COLLECTION,
        vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),  # Vektör boyutu ve mesafe metriği
        shard_number=1,                       # Dil başına 1 fiziksel shard
        sharding_method=models.ShardingMethod.CUSTOM,  # Shard-key ile özel sharding
        replication_factor=2,                 # Yedeklilik için replikasyon
    )

    # Her dil için shard-key oluştur (veri fiziksel olarak ayrılır)
    for lang in ["en", "de", "fr", "es", "ja", "zh"]:
        client.create_shard_key(settings.COLLECTION, shard_key=lang)
