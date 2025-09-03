# qdrant_streamlit_ui.py
"""Qdrant üzerinde çok dilli yıldızlı yorum arama arayüzü (Streamlit).

* 6 dil (en, es, fr, de, zh, ja)
* Dil filtresi opsiyonel
* Sonuç limiti 1‑8
* Yeni yorum eklerken PointStruct artık `id` ister → uuid4() kullanıyoruz
"""

from __future__ import annotations

import warnings
from typing import Sequence
from uuid import uuid4

import matplotlib.pyplot as plt  # noqa: F401  (Plotly bizde esas, ama ihtiyaç halinde)
import pandas as pd
import streamlit as st
from fastembed import TextEmbedding
from qdrant_client.http.models import PointStruct

from src.qdrant_setup import client
from src.config import settings

# -----------------------------------------------------------------------------
# Genel ayarlar & başlatma (YORUMLAR TÜRKÇE, ARAYÜZ İNGİLİZCE)
# -----------------------------------------------------------------------------

warnings.filterwarnings("ignore", category=DeprecationWarning, module="qdrant_client")

st.set_page_config(
    page_title="Qdrant Review Search UI",  # Arayüz başlığı İngilizce
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner=False)
def get_embedder() -> TextEmbedding:
    """Embedder nesnesini tek sefer oluşturup bellekte tutar."""
    return TextEmbedding(settings.MODEL_NAME, device=settings.DEVICE)


# Desteklenen diller
LANG_OPTS = ["en", "es", "fr", "de", "zh", "ja"]

# Her dil için grafik rengi (UI bağımsız)
LANG_COLOR = {
    "en": "#1f77b4",
    "es": "#ff7f0e",
    "fr": "#2ca02c",
    "de": "#9467bd",
    "zh": "#8c564b",
    "ja": "#e377c2",
}

# -----------------------------------------------------------------------------
# Yardımcı fonksiyonlar
# -----------------------------------------------------------------------------

def _badge(code: str) -> str:
    """Dil kodu için küçük renkli rozet üretir (Streamlit HTML)."""
    color = LANG_COLOR.get(code, "#666")
    return (
        f"<span style='background:{color};color:#fff;border-radius:4px;"
        f"padding:2px 6px;font-size:0.8rem'>{code}</span>"
    )


def query_qdrant(text: str, langs: Sequence[str], limit: int) -> pd.DataFrame:
    """Seçili shard'lar üzerinde arama yapar; **skoruna göre global ilk `limit` satırı** döner."""
    if not text:
        return pd.DataFrame()

    if not langs:
        # Dil filtresi seçilmediyse tüm dillerde ara
        langs = LANG_OPTS

    vec = next(get_embedder().embed([text]))
    collected: list[dict] = []

    for lang in langs:
        try:
            # Yıldız filtresi tamamen kaldırıldı → sade sorgu
            resp = client.query_points(
                collection_name=settings.COLLECTION,
                query=vec,
                limit=limit,               # shard başına getir
                with_payload=True,
                shard_key_selector=lang,
            )
            collected.extend(
                {
                    "language": p.payload.get("language", lang),
                    "stars": p.payload.get("stars"),
                    "score": round(p.score, 3),
                }
                for p in resp.points
            )
        except Exception as exc:
            st.error(f"Qdrant query failed for shard '{lang}': {exc}")

    if not collected:
        return pd.DataFrame()

    # Skora göre ilk N
    df_all = pd.DataFrame(collected)
    df_top = df_all.sort_values("score", ascending=False).head(limit).reset_index(drop=True)
    return df_top


def show_table(df: pd.DataFrame) -> None:
    """Sonuçları skora göre azalan şekilde tablo olarak gösterir."""
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


import plotly.express as px


def show_graphs(df: pd.DataFrame):
    """Grafikler: bar (skor) ve scatter (yıldız vs skor) — limit ≤ 8 satır."""

    tab1, tab2 = st.tabs(["Top-N score chart", "Stars × Score scatter"])

    # ---------------------------------------
    # 1) Bar grafiği – skor
    # ---------------------------------------
    with tab1:
        df_sorted = df.sort_values("score", ascending=False).reset_index(drop=True)

        fig_bar = px.bar(
            df_sorted,
            x=df_sorted.index.astype(str),
            y="score",
            color="language",
            text="score",
            title="Top results by score",
            labels={"x": "Rank", "score": "Score"},
        )

        fig_bar.update_traces(
            texttemplate="%{y:.3f}",
            textposition="outside",
            cliponaxis=False,
        )

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

    # ---------------------------------------
    # 2) Scatter – yıldız vs skor
    # ---------------------------------------
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
# Sidebar (Filtreler)
# -----------------------------------------------------------------------------

st.sidebar.header("Filters")
sel_langs = st.sidebar.multiselect("Languages (optional)", LANG_OPTS)
limit = st.sidebar.slider("Result limit", 1, 8, 8)


# -----------------------------------------------------------------------------
# Ana arayüz
# -----------------------------------------------------------------------------

st.title("🔍 Multilingual Review Search")
query = st.text_input("Enter your search query", placeholder="Great and affordable headphones…")

if st.button("Search"):
    with st.spinner("Searching…"):
        df = query_qdrant(query, sel_langs, limit)

    tab_res, tab_gfx = st.tabs(["Results", "Graphs"])
    with tab_res:
        show_table(df)
        if not df.empty:
            st.download_button("Download CSV", df.to_csv(index=False).encode(), "results.csv", "text/csv")
    with tab_gfx:
        show_graphs(df)


# -----------------------------------------------------------------------------
# Yeni yorum ekle (backend aynı, arayüz İngilizce)
# -----------------------------------------------------------------------------

with st.expander("Add new review to database"):
    new_text = st.text_input("Review text")
    c1, c2 = st.columns(2)
    with c1:
        new_lang = st.selectbox("Language", LANG_OPTS)
    with c2:
        new_star = st.selectbox("Stars", [1, 2, 3, 4, 5], index=4)

    if st.button("Save to DB") and new_text:
        vec = next(get_embedder().embed([new_text]))
        point = PointStruct(
            id=str(uuid4()),
            vector=vec,
            payload={"language": new_lang, "stars": new_star},
        )
        client.upsert(settings.COLLECTION, [point], shard_key_selector=new_lang)
        st.success("Review added!")


st.caption("Built with Streamlit • Powered by FastEmbed & Qdrant")
