# src/config.py
# Proje genelinde ortam değişkenlerini ve model ayarlarını merkezi olarak yöneten yapılandırma dosyası.

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import AnyHttpUrl


class Settings(BaseSettings):
    # Qdrant bağlantı ayarları
    QDRANT_URL: AnyHttpUrl
    QDRANT_API_KEY: str
    COLLECTION: str = "amazon_reviews_multi"

    # Embedding modeli ayarları
    MODEL_NAME: str = "BAAI/bge-small-en-v1.5"
    DEVICE: str = "cuda"

    # pydantic-settings yapılandırması
    model_config = SettingsConfigDict(
        env_file=".env",                # Ortam değişkenlerini .env dosyasından oku
        case_sensitive=True,             # Değişken adlarında büyük/küçük harf duyarlılığı
        extra="ignore",                # Tanımsız ortam değişkenlerini yoksay extra="forbid" derseniz beklenmeyen değişkenler ValidationError fırlatır;
                                       # güvenlik sıkılaştırmak istediğinizde tercih edebilirsiniz.
    )


# --------------  BU SATIR ÇOK ÖNEMLİ  --------------
settings = Settings()         #  Dışa aktarılan ve projede her yerde kullanılan ayar instance'ı











"""
Neden BaseSettings, klasik BaseModel değil?
Özellik	                           BaseModel	                                                   BaseSettings
Kaynak	Yalnızca doğrudan Python argümanları veya dict’ler	                                  Python argümanları + ortam değişkenleri (ENV, .env, Docker secrets…)
Kullanım Senaryosu	API request/response şemaları, iş kuralları, veri doğrulama	              Uygulama konfigürasyonu, gizli anahtarlar, bağlantı dizeleri
Yükleme Sırası	Instantiation sırasında verilen değerler	                                 1 Fonksiyon argümanları → 2 Ort. değişkenleri → 3 Varsayılanlar
Gizli Bilgi Yönetimi	Harici desteğe ihtiyaç duyar	                                      .env / runtime ortamından güvenli çekme yerleşik

Bu projede QDRANT_API_KEY, QDRANT_URL gibi değerler çalıştığınız makineye göre sürekli değişecek ve git deposuna yazılmaması gereken gizli anahtarlar.
BaseSettings bunları otomatik okuyup doğrular; ayrıca type-hint’ler sayesinde IDE tamamlaması ve statik analiz (mypy, Ruff) sorunsuz çalışır.
"""