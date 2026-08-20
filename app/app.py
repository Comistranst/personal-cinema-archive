from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    catalog = pd.read_csv(ROOT / "data" / "public_catalog.csv", dtype={"review_id": str})
    embeddings = np.load(ROOT / "model" / "embeddings.npy")
    mapping = pd.read_csv(ROOT / "model" / "public_mapping.csv", dtype={"review_id": str})
    if len(catalog) != len(embeddings) or len(mapping) != len(embeddings):
        raise ValueError("Public catalog, mapping, and embeddings must have the same length.")
    return catalog, embeddings, mapping


@st.cache_resource
def load_model() -> SentenceTransformer:
    snapshots = ROOT / "model" / "huggingface" / "models--sentence-transformers--paraphrase-multilingual-mpnet-base-v2" / "snapshots"
    local = sorted(snapshots.glob("*"))
    return SentenceTransformer(str(local[-1]), local_files_only=True) if local else SentenceTransformer(MODEL_NAME)


def match_level(score: float, references: np.ndarray) -> tuple[str, str]:
    low, high = np.quantile(references, [0.33, 0.67])
    if score >= high:
        return "\u9ad8\u5339\u914d", "\u4e0e\u4f60\u7684\u9ad8\u5206\u5f71\u8bc4\u5728\u8bed\u4e49\u4e0a\u5341\u5206\u63a5\u8fd1\u3002"
    if score >= low:
        return "\u4e2d\u5339\u914d", "\u4e0e\u4f60\u504f\u7231\u7684\u4e3b\u9898\u548c\u60c5\u7eea\u5b58\u5728\u90e8\u5206\u4ea4\u96c6\u3002"
    return "\u4f4e\u5339\u914d", "\u4e0e\u5f53\u524d\u9ad8\u5206\u6837\u672c\u7684\u8bed\u4e49\u8ddd\u79bb\u8f83\u8fdc\uff1b\u8fd9\u4e0d\u4ee3\u8868\u4f5c\u54c1\u8d28\u91cf\u3002"


def infer_reasons(text: str) -> list[str]:
    groups = {
        "\u4eba\u7269\u5fc3\u7406\u4e0e\u5185\u5fc3\u53d8\u5316": ["\u5fc3\u7406", "\u5185\u5fc3", "\u521b\u4f24", "\u5b64\u72ec", "\u4eba\u7269"],
        "\u514b\u5236\u800c\u6d53\u70c8\u7684\u60c5\u611f\u8868\u8fbe": ["\u60c5\u611f", "\u7231\u60c5", "\u5173\u7cfb", "\u5931\u53bb", "\u4eb2\u60c5"],
        "\u57ce\u5e02\u3001\u793e\u4f1a\u4e0e\u73b0\u5b9e\u89c2\u5bdf": ["\u57ce\u5e02", "\u793e\u4f1a", "\u73b0\u5b9e", "\u9636\u7ea7", "\u65f6\u4ee3", "\u751f\u6d3b"],
        "\u4f5c\u8005\u98ce\u683c\u4e0e\u955c\u5934\u8bed\u8a00": ["\u4f5c\u8005", "\u955c\u5934", "\u6444\u5f71", "\u8272\u5f69", "\u7f8e\u5b66", "\u827a\u672f"],
    }
    return [name for name, terms in groups.items() if any(term in text for term in terms)] or ["\u8bed\u4e49\u7279\u5f81\u4e0e\u4e2a\u4eba\u9ad8\u5206\u7247\u5355\u63a5\u8fd1"]


def style() -> None:
    st.markdown("""<style>
    .stApp{background:#fff65b;background-image:radial-gradient(circle at 90% 8%,#ff5dce 0 13%,transparent 13.2%),radial-gradient(circle at 6% 82%,#00d9ff 0 16%,transparent 16.2%);color:#181818}.block-container{max-width:1160px;padding-top:2.3rem;padding-bottom:4rem}header[data-testid="stHeader"]{background:rgba(255,246,91,.78)}#MainMenu,footer{visibility:hidden}h1,h2{letter-spacing:-.06em;color:#181818}.eyebrow,.archive-label{color:#6b23f4;font:500 11px monospace;letter-spacing:.14em;text-transform:uppercase}.hero{background:#ff5dce;border:4px solid #181818;box-shadow:10px 10px 0 #181818;padding:28px 32px 32px;margin:8px 0 30px;transform:rotate(-1deg)}.hero h1{font-size:clamp(3rem,8vw,6.4rem);line-height:.9;margin:10px 0 16px;color:#181818;text-shadow:3px 3px 0 #fff65b}.hero p{color:#181818;max-width:530px;font-weight:600;line-height:1.75;margin:0}.stat{background:#fff;border:3px solid #181818;box-shadow:5px 5px 0 #181818;padding:14px 16px;margin:14px 8px 36px 0;transform:rotate(1deg)}.stat-num{font:500 30px monospace;color:#6b23f4;line-height:1.1}.stat-label{color:#181818;font-size:12px;font-weight:700;margin-top:7px}.stTabs [data-baseweb="tab-list"]{border-bottom:3px solid #181818;gap:10px}.stTabs [data-baseweb="tab"]{background:#fff;color:#181818;border:2px solid #181818;border-bottom:0;font:500 12px monospace;padding:12px 17px}.stTabs [aria-selected="true"]{background:#00d9ff!important;border-bottom:3px solid #00d9ff!important}.stTextInput input,.stTextArea textarea{background:#fff!important;border:3px solid #181818!important;color:#181818!important;border-radius:0!important}.stButton button{background:#6b23f4!important;color:#fff65b!important;border:3px solid #181818!important;box-shadow:5px 5px 0 #181818!important;border-radius:0!important;font-weight:700!important;padding:.75rem 1.15rem!important}.report{margin-top:32px;padding:28px;border:4px solid #181818;background:#00d9ff;box-shadow:10px 10px 0 #6b23f4}.report-title{font:700 11px monospace;letter-spacing:.13em;color:#6b23f4}.report-level{font-size:3.4rem;font-weight:900;color:#181818;margin:7px 0}.report-copy{color:#181818;font-weight:600;line-height:1.7}.match-row{display:flex;justify-content:space-between;gap:20px;border:3px solid #181818;background:#fff;padding:13px 16px;color:#181818;margin:9px 0;box-shadow:4px 4px 0 #ff5dce}.match-meta{color:#6b23f4;font:700 11px monospace;white-space:nowrap}[data-testid="stDataFrame"]{border:3px solid #181818;background:#fff}</style>""", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="\u79c1\u4eba\u89c2\u5f71\u6863\u6848", page_icon="\u25a0", layout="wide")
    style()
    catalog, embeddings, mapping = load_data()
    ratings = catalog["rating"].astype(float)
    high_mask = mapping["rating"].astype(float).ge(9).to_numpy()
    high_embeddings = embeddings[high_mask]
    st.markdown("<section class='hero'><div class='eyebrow'>Personal cinema archive / 2024—2026</div><h1>\u79c1\u4eba\u89c2\u5f71<br>\u6863\u6848</h1><p>\u4e0d\u662f\u901a\u7528\u63a8\u8350\uff0c\u800c\u662f\u4e00\u4efd\u6839\u636e\u4e2a\u4eba\u89c2\u5f71\u8bb0\u5f55\u751f\u6210\u7684\u79c1\u4eba\u770b\u7247\u6307\u5357\u3002</p></section>", unsafe_allow_html=True)
    for column, number, label in zip(st.columns(3), [str(len(catalog)), f"{ratings.mean():.1f}", str(int(ratings.eq(10).sum()))], ["\u6761\u89c2\u5f71\u8bb0\u5f55", "\u5e73\u5747\u8bc4\u5206 / 10", "\u6ee1\u5206\u4f5c\u54c1"], strict=True):
        column.markdown(f"<div class='stat'><div class='stat-num'>{number}</div><div class='stat-label'>{label}</div></div>", unsafe_allow_html=True)
    recommend, profile, library = st.tabs(["01 / \u8bd5\u6620\u5ba4", "02 / \u5ba1\u7f8e\u6863\u6848", "03 / \u7247\u5355\u7d22\u5f15"])
    with recommend:
        st.markdown("<p class='archive-label'>Submit a film for consideration</p><h2>\u5b83\u9002\u5408\u4f60\u5417\uff1f</h2>", unsafe_allow_html=True)
        title = st.text_input("\u7247\u540d", placeholder="\u4f8b\u5982\uff1a\u9a7e\u9a76\u6211\u7684\u8f66")
        description = st.text_area("\u5267\u60c5\u3001\u4e3b\u9898\uff0c\u6216\u4f60\u60f3\u770b\u5b83\u7684\u539f\u56e0", placeholder="\u4f8b\u5982\uff1a\u6162\u8282\u594f\u3001\u4eba\u7269\u5173\u7cfb\u4e0e\u60c5\u611f\u521b\u4f24\u3002", height=150)
        if st.button("\u751f\u6210\u5339\u914d\u62a5\u544a"):
            if not description.strip():
                st.warning("\u8bf7\u81f3\u5c11\u8f93\u5165\u5267\u60c5\u7b80\u4ecb\u3001\u4e3b\u9898\u6216\u89c2\u5f71\u7406\u7531\u3002")
            else:
                query = load_model().encode([f"{title}\n{description}"], normalize_embeddings=True)[0]
                similarities = embeddings @ query
                score = float((high_embeddings @ query).mean())
                level, explanation = match_level(score, (high_embeddings @ high_embeddings.T).mean(axis=1))
                st.markdown(f"<section class='report'><div class='report-title'>PERSONAL MATCH · {score:.3f}</div><div class='report-level'>{level}</div><p class='report-copy'>{explanation}<br><br>\u8bc6\u522b\u5230\u7684\u9760\u8fd1\u4e3b\u9898\uff1a{' / '.join(infer_reasons(description))}</p></section>", unsafe_allow_html=True)
                for index in [i for i in np.argsort(similarities)[::-1] if high_mask[i]][:5]:
                    row = mapping.iloc[index]
                    st.markdown(f"<div class='match-row'><span>\u300a{row['title']}\u300b</span><span class='match-meta'>{float(row['rating']):g}/10 · {similarities[index]:.3f}</span></div>", unsafe_allow_html=True)
    with profile:
        st.markdown("<p class='archive-label'>What the record suggests</p><h2>\u5ba1\u7f8e\u6863\u6848</h2>", unsafe_allow_html=True)
        st.dataframe(catalog.loc[ratings.ge(9), ["title", "rating"]].sort_values("rating", ascending=False), width="stretch", hide_index=True)
    with library:
        st.markdown("<p class='archive-label'>The public-facing collection</p><h2>\u7247\u5355\u7d22\u5f15</h2>", unsafe_allow_html=True)
        st.caption("\u6b64\u516c\u5f00\u7248\u672c\u4e0d\u5c55\u793a\u539f\u59cb\u5f71\u8bc4\u3001\u6e05\u6d17\u6587\u672c\u6216\u89c2\u770b\u65e5\u671f\u3002")
        st.dataframe(catalog[["review_id", "title", "rating"]], width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
