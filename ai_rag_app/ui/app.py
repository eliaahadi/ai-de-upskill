from __future__ import annotations

# from pathlib import Path
import pandas as pd
import streamlit as st

from ai_rag_app.src.config import DOCS_DIR, VSTORE_DIR
from ai_rag_app.src.index_docs import build_index
from ai_rag_app.src.rag_chain import answer
from ai_rag_app.src.retriever import get_collection, retrieve


st.set_page_config(page_title="RAG Playground", layout="wide")
st.title("RAG playground")

# ---- Sidebar: index & docs controls ----
st.sidebar.header("Index control")
st.sidebar.write(f"Docs dir: `{DOCS_DIR}`")
st.sidebar.write(f"Vector store: `{VSTORE_DIR}`")

uploaded = st.sidebar.file_uploader(
    "Add docs (.md/.txt/.pdf)", type=["md", "markdown", "txt", "pdf"], accept_multiple_files=True
)
if uploaded:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for f in uploaded:
        dest = DOCS_DIR / f.name
        with open(dest, "wb") as out:
            out.write(f.read())
    st.sidebar.success(f"Saved {len(uploaded)} file(s) to {DOCS_DIR}")

if st.sidebar.button("Re-index now"):
    with st.spinner("Indexing docs..."):
        build_index(DOCS_DIR, VSTORE_DIR)
    st.sidebar.success("Index rebuilt")

# ---- Ask panel ----
q = st.text_input("Ask a question about your docs")
cols = st.columns([1, 1, 2, 2])
with cols[0]:
    k = st.slider("Top-k", 1, 10, 5)
with cols[1]:
    with_eval = st.checkbox("Compute eval", value=True)

if st.button("Ask"):
    col = get_collection()
    if col.count() == 0:
        st.error("Vector store is empty. Add docs and re-index first.")
    else:
        with st.spinner("Retrieving and answering..."):
            res = answer(q, k=k, mode="extractive", with_eval=with_eval)

        st.subheader("Answer")
        st.write(res["answer"])

        # metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Retrieved", res.get("retrieved", 0))
        m2.metric("Context chars", res.get("context_chars", 0))
        if with_eval and "eval" in res:
            m3.metric("Support rate", f"{res['eval']['support_rate']:.2f}")
            m4.metric("Q↔Ctx cosine", f"{res['eval']['q_ctx_cosine']:.2f}")

        # sources
        st.markdown("### Sources")
        if res.get("sources"):
            src_df = pd.DataFrame(res["sources"])
            st.dataframe(src_df)
        else:
            st.info("No sources returned.")

        # retrieved chunks preview
        with st.expander("See retrieved chunks"):
            hits = retrieve(q, k=k)
            for doc, meta in hits:
                st.markdown(
                    f"**{meta.get('source','?')}** • chunk {meta.get('chunk_index')} • dist {meta.get('distance', 0):.3f}"
                )
                st.code(doc[:1200], language="markdown")
