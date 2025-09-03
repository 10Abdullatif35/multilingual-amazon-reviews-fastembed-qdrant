# qdrant_streamlit_ui.py
"""Streamlit UI for multilingual star‑rated review search on Qdrant.

* 6 dil (en, es, fr, de, zh, ja)
* Dil ve yıldız filtreleri opsiyonel
* Sonuç limiti 1‑8
* Yeni yorum eklerken PointStruct artık `id` ister → uuid4() kullanıyoruz
"""

# -----------------------------------------------------------------------------
#  Ö N E M L İ  N O T
# -----------------------------------------------------------------------------
# (TR) Bu dosyadaki kod mantığına dokunulmamıştır.  Sadece kritik bölümlere
#      Türkçe açıklama satırları (# ...) eklenmiştir.  Fonksiyon isimleri ve
#      değişkenler aynı kalmıştır, dolayısıyla orijinal davranış korunur.
# -----------------------------------------------------------------------------

from __future__ import annotations

import warnings
from typing import Sequence
from uuid import uuid4

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from fastembed import TextEmbedding
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, PointStruct

from src.qdrant_setup import client
from src.config import settings

# -----------------------------------------------------------------------------
# Config & init
# -----------------------------------------------------------------------------

# (TR) Qdrant istemcisinden gelen DeprecationWarning mesajlarını bastırıyoruz.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="qdrant_client")

# (TR) Streamlit sayfa yapılandırması: başlık, geniş düzen ve kenar çubuğu ayarı
st.set_page_config(
    page_title="Qdrant Review Search UI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
# (TR) Embedder nesnesini bellekte önbelleğe alıyoruz; tekrar çağrıldığında
#      yeniden yüklenmesini engeller, performans kazandırır.
#      TextEmbedding: FastEmbed kitaplığından çok‑dilli gömme (embedding) modeli.

def get_embedder() -> TextEmbedding:
    return TextEmbedding(settings.MODEL_NAME, device=settings.DEVICE)

# (TR) Kullanıcıya sunulacak sabit dil ve yıldız seçenekleri
LANG_OPTS = ["en", "es", "fr", "de", "zh", "ja"]
STAR_OPTS = [1, 2, 3, 4, 5]

# (TR) Grafiklerde ve rozetlerde kullanılacak renk eşlemeleri
LANG_COLOR = {
    "en": "#1f77b4",
    "es": "#ff7f0e",
    "fr": "#2ca02c",
    "de": "#9467bd",
    "zh": "#8c564b",
    "ja": "#e377c2",
}

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _badge(code: str) -> str:
    """(TR) Dil kodu için renkli HTML rozet döndürür."""
    color = LANG_COLOR.get(code, "#666")
    return (
        f"<span style='background:{color};color:#fff;border-radius:4px;"
        f"padding:2px 6px;font-size:0.8rem'>{code}</span>"
    )


# -----------------------------------------------------------------------------
# Ana arama fonksiyonu
# -----------------------------------------------------------------------------


def query_qdrant(text: str, langs: Sequence[str], stars: Sequence[int], limit: int) -> pd.DataFrame:
    """(TR) Çok‑dilli shard’lar üzerinde arama yapar ve **skora göre** en iyi
    `limit` satırı döndürür.

    Parametreler
    ------------
    text   : Aranacak metin (zorunlu)
    langs  : Dil filtre listesi (boş ise tüm diller)
    stars  : Yıldız filtre listesi (boş ise tüm yıldızlar)
    limit  : Sonuç sayısı (1‑8)
    """

    if not text:
        return pd.DataFrame()

    # (TR) Hiç dil seçilmediyse tüm dilleri ara
    if not langs:
        langs = LANG_OPTS
    star_filter_flag = bool(stars)

    # (TR) Sorgu metnini embedding vektörüne dönüştür
    vec = next(get_embedder().embed([text]))
    collected: list[dict] = []
    for lang in langs:
        q_filter: Filter | None = None
        if star_filter_flag:
            # (TR) Yıldız filtresi: OR (should) koşulları
            q_filter = Filter(
                should=[FieldCondition(key="stars", match=MatchValue(value=s)) for s in stars]
            )
        try:
            resp = client.query_points(
                collection_name=settings.COLLECTION,
                query=vec,
                limit=limit,  # (TR) Her shard için en fazla `limit` kayıt çek
                with_payload=True,
                shard_key_selector=lang,
                query_filter=q_filter,
            )
            # (TR) Her bir nokta için gerekli alanları topla
            collected.extend(
                {
                    "language": p.payload.get("language", lang),
                    "stars": p.payload.get("stars"),
                    "score": round(p.score, 3),
                }
                for p in resp.points
            )
        except Exception as exc:
            st.error(f"Qdrant query failed for '{lang}': {exc}")

    if not collected:
        return pd.DataFrame()

    # (TR) Tüm shard’lardan gelen kayıtlar → en yüksek skorluları seç
    df_all = pd.DataFrame(collected)
    df_top = df_all.sort_values("score", ascending=False).head(limit).reset_index(drop=True)
    return df_top


# -----------------------------------------------------------------------------
# Sonuçları tablo halinde gösterme
# -----------------------------------------------------------------------------

def show_table(df: pd.DataFrame) -> None:
    """(TR) DataFrame'i skora göre azalan sıralayıp Streamlit tablosu olarak gösterir."""

    if df.empty:
        st.info("No results.")
        return

    df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

    st.dataframe(
        df_sorted,
        use_container_width=True,
        hide_index=True,
        column_config={
            "language": st.column_config.Column("Lang", width="small"),
            "stars": st.column_config.NumberColumn("★"),
            "score": st.column_config.NumberColumn("Score", format="%.3f"),
        },
    )


# -----------------------------------------------------------------------------
# Grafikler (Plotly)
# -----------------------------------------------------------------------------

import plotly.express as px
import pandas as pd
import streamlit as st


def show_graphs(df: pd.DataFrame):
    """(TR) df: ['language', 'stars', 'score'] sütunlarına sahip; maksimum 8 satır."""

    tab1, tab2 = st.tabs(["Top-N score chart", "Stars × Score scatter"])

    # --------------------------------------------------
    # 1) Bar chart – her sonuç için tek sütun
    # --------------------------------------------------
    with tab1:
        # (TR) Sonuçları skora göre sırala → bar grafiği
        df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

        fig_bar = px.bar(
            df_sorted,
            x=df_sorted.index.astype(str),  # 0,1,2…  (kategorik)
            y="score",
            color="language",
            text="score",
            title="Top results by score",
            labels={"x": "Rank", "score": "Score"},
        )

        # (TR) Her bar’ın üzerine skor metni
        fig_bar.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
            cliponaxis=False,
        )

        # (TR) Y‑ekseni 0‑1 arası sabit ölçek
        fig_bar.update_yaxes(
            range=[0, 1],
            dtick=0.25,
            title="Score",
        )

        fig_bar.update_xaxes(
            tickvals=df_sorted.index,
            ticktext=(df_sorted.index + 1).astype(str),
            title="Rank",
        )

        fig_bar.update_layout(
            legend_title_text="Language",
            bargap=0.25,
        )

        st.plotly_chart(fig_bar, use_container_width=True)

    # --------------------------------------------------
    # 2) Scatter – yıldız (x) ve skor (y)
    # --------------------------------------------------
    with tab2:
        fig_sc = px.scatter(
            df,
            x="stars",
            y="score",
            color="language",
            hover_data=["language", "stars", "score"],
            title="Stars vs. Score",
        )
        fig_sc.update_xaxes(dtick=1, title="Stars (★)")
        fig_sc.update_yaxes(title="Score")
        fig_sc.update_layout(legend_title_text="Language")
        st.plotly_chart(fig_sc, use_container_width=True)


# -----------------------------------------------------------------------------
# Sidebar (Kenar Çubuğu)
# -----------------------------------------------------------------------------

st.sidebar.header("Filters")
# (TR) Çoklu seçimler: diller ve yıldızlar
sel_langs = st.sidebar.multiselect("Languages (optional)", LANG_OPTS)
sel_stars = st.sidebar.multiselect("Stars (optional)", STAR_OPTS)
# (TR) Kaydırıcı ile sonuç limiti
limit = st.sidebar.slider("Result limit", 1, 8, 8)


# -----------------------------------------------------------------------------
# Main interface (Ana Arayüz)
# -----------------------------------------------------------------------------

st.title("🔍 Multilingual Review Search")
# (TR) Sorgu girişi
query = st.text_input("Enter your search query", placeholder="Great and affordable headphones…")

# (TR) Arama butonu – tıklandığında sorgu çalışır
if st.button("Search"):
    with st.spinner("Searching…"):
        df = query_qdrant(query, sel_langs, sel_stars, limit)

    tab_res, tab_gfx = st.tabs(["Results", "Graphs"])
    with tab_res:
        show_table(df)
        if not df.empty:
            # (TR) Sonuçları CSV olarak indir
            st.download_button("Download CSV", df.to_csv(index=False).encode(), "results.csv", "text/csv")
    with tab_gfx:
        show_graphs(df)

# -----------------------------------------------------------------------------
# Yeni yorum ekleme (Veritabanına)
# -----------------------------------------------------------------------------

with st.expander("Add new review to database"):
    new_text = st.text_input("Review text")
    c1, c2 = st.columns(2)
    with c1:
        new_lang = st.selectbox("Language", LANG_OPTS)
    with c2:
        new_star = st.selectbox("Stars", STAR_OPTS, index=4)

    # (TR) Kaydet butonu: yeni embed ve Qdrant upsert
    if st.button("Save to DB") and new_text:
        vec = next(get_embedder().embed([new_text]))
        point = PointStruct(
            id=str(uuid4()),  # (TR) Her yorum için benzersiz UUID
            vector=vec,
            payload={"language": new_lang, "stars": new_star},
        )
        client.upsert(settings.COLLECTION, [point], shard_key_selector=new_lang)
        st.success("Review added!")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.caption("Built with Streamlit • Powered by FastEmbed & Qdrant")
