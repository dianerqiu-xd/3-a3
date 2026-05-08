from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from semantic_core import (
    DEFAULT_CORPUS,
    TinyAnalogyVectors,
    analogy_table,
    average_sentence_vector,
    compute_lsa_coordinates,
    compute_tfidf,
    cosine_similarity,
    most_similar_from_vector,
    normalized_documents,
    tokenized_corpus,
    top_similar,
    train_fasttext,
    train_word2vec,
    word_pair_similarity,
)


st.set_page_config(page_title="A3 Semantic Analysis Lab", page_icon="NLP", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.25rem; padding-bottom: 2rem; }
      [data-testid="stMetricValue"] { font-size: 1.55rem; }
      .soft-note {
        color: #475569;
        font-size: 0.92rem;
        line-height: 1.55;
      }
      .small-title {
        font-weight: 700;
        color: #0f172a;
        margin: 0.35rem 0 0.15rem;
      }
      .stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def cached_tfidf(corpus: str):
    return compute_tfidf(corpus)


@st.cache_data(show_spinner=False)
def cached_lsa(corpus: str, matrix_type: str, max_features: int):
    return compute_lsa_coordinates(corpus, matrix_type=matrix_type, max_features=max_features)


@st.cache_resource(show_spinner=False)
def cached_word2vec(corpus: str, sg: int, window: int, vector_size: int, epochs: int):
    return train_word2vec(tokenized_corpus(corpus), sg=sg, window=window, vector_size=vector_size, epochs=epochs)


@st.cache_resource(show_spinner=False)
def cached_fasttext(corpus: str, window: int, vector_size: int, epochs: int):
    return train_fasttext(tokenized_corpus(corpus), window=window, vector_size=vector_size, epochs=epochs)


@st.cache_resource(show_spinner=False)
def load_glove_vectors():
    try:
        import gensim.downloader as api

        return api.load("glove-twitter-25"), "GloVe Twitter 25d"
    except Exception as exc:
        return TinyAnalogyVectors(), f"内置演示向量（GloVe 下载失败：{type(exc).__name__}）"


def safe_word_options(model) -> list[str]:
    words = sorted(model.wv.key_to_index.keys())
    preferred = ["computer", "model", "data", "language", "music", "city"]
    return [word for word in preferred if word in words] + [word for word in words if word not in preferred]


with st.sidebar:
    st.header("语料输入")
    corpus = st.text_area(
        "英文语料（建议 500-1000 词）",
        value=DEFAULT_CORPUS.strip(),
        height=380,
    )
    docs = normalized_documents(corpus)
    tokens = tokenized_corpus(corpus)
    token_count = sum(len(sentence) for sentence in tokens)
    vocab_count = len({token for sentence in tokens for token in sentence})
    st.metric("句子数", len(docs))
    st.metric("词元数", token_count)
    st.metric("词表规模", vocab_count)
    st.caption("四个标签页共用这段语料，便于观察不同语义表示模型的差异。")


st.title("A3 语义表示与对比分析系统")
st.markdown(
    '<div class="soft-note">集成 TF-IDF / LSA、Word2Vec、GloVe、FastText 与简单 Sent2Vec。每个模块都可以直接交互测试课件中的核心思想。</div>',
    unsafe_allow_html=True,
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["1. TF-IDF 与 LSA", "2. Word2Vec", "3. GloVe 类比", "4. FastText & Sent2Vec"]
)

with tab1:
    st.subheader("传统统计模型：TF-IDF 与 LSA")
    try:
        tfidf = cached_tfidf(corpus)
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.markdown('<div class="small-title">Top 5 TF-IDF 关键词</div>', unsafe_allow_html=True)
            st.dataframe(tfidf.top_keywords, use_container_width=True, hide_index=True)
        with col_b:
            st.markdown('<div class="small-title">句子切分后的文档集合</div>', unsafe_allow_html=True)
            st.dataframe(pd.DataFrame({"document": tfidf.documents}), use_container_width=True, hide_index=True)

        with st.expander("查看 TF-IDF 矩阵", expanded=False):
            st.dataframe(tfidf.matrix.round(3), use_container_width=True)

        st.divider()
        col_l, col_r = st.columns([0.28, 0.72])
        with col_l:
            matrix_type = st.radio("LSA 输入矩阵", ["tfidf", "count"], format_func=lambda x: "TF-IDF" if x == "tfidf" else "One-hot / Count")
            max_features = st.slider("参与降维的词汇数", 20, 100, 60, 10)
        lsa_df = cached_lsa(corpus, matrix_type, max_features)
        with col_r:
            st.scatter_chart(lsa_df, x="x", y="y", color="#2563eb", size=70, use_container_width=True, height=430)
            st.dataframe(lsa_df.sort_values("word"), use_container_width=True, hide_index=True)
        st.info("观察提示：如果两个词经常出现在相似句子中，它们在 LSA 二维空间中通常会靠得更近。")
    except Exception as exc:
        st.error(f"模块 1 暂时无法计算：{exc}")

with tab2:
    st.subheader("Word2Vec 实时训练：CBOW vs Skip-Gram")
    col_cfg, col_out = st.columns([0.34, 0.66])
    with col_cfg:
        architecture = st.radio("训练架构", ["CBOW", "Skip-Gram"], horizontal=True)
        sg = 0 if architecture == "CBOW" else 1
        window = st.slider("window 上下文窗口", 2, 10, 5)
        vector_size = st.slider("向量维度", 20, 100, 50, 10)
        epochs = st.slider("训练轮数", 40, 200, 120, 20)
    try:
        with st.spinner("正在训练 Word2Vec..."):
            w2v = cached_word2vec(corpus, sg=sg, window=window, vector_size=vector_size, epochs=epochs)
        with col_cfg:
            options = safe_word_options(w2v)
            default_word = options[0] if options else ""
            query_word = st.text_input("查询相似词", value=default_word)
            selected = st.selectbox("或从当前词表选择", options=options, index=0 if options else None)
            if query_word.strip() == "" and selected:
                query_word = selected
        with col_out:
            st.metric("当前架构", architecture)
            st.metric("训练词表", len(w2v.wv.key_to_index))
            try:
                similar = top_similar(w2v, query_word, topn=5)
                st.markdown('<div class="small-title">Top 5 余弦相似词</div>', unsafe_allow_html=True)
                st.dataframe(similar, use_container_width=True, hide_index=True)
            except KeyError:
                st.warning(f"`{query_word}` 不在当前小语料词表中，请换一个词。")
        st.info("观察提示：切换 CBOW 和 Skip-Gram 后，同一个词的 Top 5 结果可能会变化，尤其是在小语料上更明显。")
    except Exception as exc:
        st.error(f"模块 2 暂时无法训练：{exc}")

with tab3:
    st.subheader("预训练 GloVe：词类比与词义相似度")
    vectors, vector_name = load_glove_vectors()
    st.caption(f"当前向量来源：{vector_name}")

    col_ana, col_sim = st.columns(2)
    with col_ana:
        st.markdown('<div class="small-title">Word Analogy: A - B + C</div>', unsafe_allow_html=True)
        word_a = st.text_input("A", value="king")
        word_b = st.text_input("B", value="man")
        word_c = st.text_input("C", value="woman")
        try:
            result = analogy_table(vectors, word_a, word_b, word_c, topn=5)
            st.dataframe(result, use_container_width=True, hide_index=True)
        except KeyError as exc:
            st.warning(f"词表中没有：{exc}")
        st.caption("可试：paris - france + china；tokyo - japan + france。")

    with col_sim:
        st.markdown('<div class="small-title">两个单词的语义相似度</div>', unsafe_allow_html=True)
        sim_a = st.text_input("单词 1", value="computer")
        sim_b = st.text_input("单词 2", value="algorithm")
        try:
            score = word_pair_similarity(vectors, sim_a, sim_b)
            st.metric("Cosine Similarity", f"{score:.4f}")
        except KeyError as exc:
            st.warning(f"词表中没有：{exc}")
    st.info("观察提示：GloVe 使用全局共现统计学习向量，经典类比可以检验向量空间是否捕捉到了线性语义关系。")

with tab4:
    st.subheader("FastText 子词特征与句向量")
    col_cfg, col_oov = st.columns([0.34, 0.66])
    with col_cfg:
        ft_window = st.slider("FastText window", 2, 10, 5)
        ft_vector_size = st.slider("FastText 向量维度", 20, 100, 50, 10)
        ft_epochs = st.slider("FastText 训练轮数", 40, 200, 120, 20)
        oov_word = st.text_input("OOV / 拼写错误测试词", value="computeer")
    try:
        with st.spinner("正在训练 FastText 与对照 Word2Vec..."):
            ft_model = cached_fasttext(corpus, window=ft_window, vector_size=ft_vector_size, epochs=ft_epochs)
            w2v_for_oov = cached_word2vec(corpus, sg=1, window=ft_window, vector_size=ft_vector_size, epochs=ft_epochs)

        with col_oov:
            st.markdown('<div class="small-title">OOV 对比</div>', unsafe_allow_html=True)
            try:
                _ = w2v_for_oov.wv[oov_word.lower()]
                st.success("Word2Vec 找到了这个词。")
            except KeyError:
                st.warning("Word2Vec：未登录词（KeyError），无法直接提取向量。")

            try:
                ft_vector = ft_model.wv[oov_word.lower()]
                st.success(f"FastText：成功通过字符 n-gram 生成向量，维度 {len(ft_vector)}。")
                st.dataframe(most_similar_from_vector(ft_model, ft_vector, topn=5), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(f"FastText OOV 计算失败：{exc}")

        st.divider()
        st.markdown('<div class="small-title">简单 Sent2Vec：平均池化句向量</div>', unsafe_allow_html=True)
        sent_a = st.text_area(
            "句子 A",
            value="A computer program learns useful patterns from data and makes predictions for new examples.",
            height=90,
        )
        sent_b = st.text_area(
            "句子 B",
            value="An intelligent model studies examples, discovers structure, and predicts future observations.",
            height=90,
        )
        vec_a = average_sentence_vector(sent_a, ft_model.wv)
        vec_b = average_sentence_vector(sent_b, ft_model.wv)
        score = cosine_similarity(vec_a, vec_b)
        if score is None:
            st.warning("至少有一个句子没有可用词向量，请换一组英文句子。")
        else:
            st.metric("两个句向量的余弦相似度", f"{score:.4f}")
        st.info("观察提示：平均词向量的 Sent2Vec 方法简单直观，但会忽略词序和句法结构。")
    except Exception as exc:
        st.error(f"模块 4 暂时无法训练：{exc}")
