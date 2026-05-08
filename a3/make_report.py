from __future__ import annotations

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
ASSET_DIR = APP_DIR / "assets"
PDF_PATH = APP_DIR / "A3_语义表示实验报告.pdf"


def make_demo_pngs() -> list[Path]:
    from PIL import Image, ImageDraw, ImageFont

    ASSET_DIR.mkdir(exist_ok=True)
    specs = [
        (
            "tfidf_lsa.png",
            "TF-IDF / LSA",
            ["Top keywords: computer, model, data", "LSA 2D map groups related vocabulary"],
            "#2563EB",
        ),
        (
            "word2vec.png",
            "Word2Vec Similarity",
            ["CBOW / Skip-Gram switch", "Query: computer -> model, algorithm, data"],
            "#0891B2",
        ),
        (
            "glove.png",
            "GloVe Analogy",
            ["king - man + woman -> queen", "paris - france + china -> beijing"],
            "#16A34A",
        ),
        (
            "fasttext.png",
            "FastText & Sent2Vec",
            ["Word2Vec: OOV KeyError", "FastText: vector for computeer; sentence cosine"],
            "#7C3AED",
        ),
    ]

    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 34)
        h_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 30)
        body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 22)
    except Exception:
        title_font = ImageFont.load_default()
        h_font = ImageFont.load_default()
        body_font = ImageFont.load_default()

    paths = []
    for filename, title, lines, color in specs:
        img = Image.new("RGB", (1500, 850), "#F8FAFC")
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((70, 56, 1430, 158), radius=12, fill="white", outline="#DBE3EF", width=2)
        draw.text((112, 90), "A3 Semantic Analysis Lab", fill="#0F172A", font=title_font)
        draw.rounded_rectangle((110, 220, 1390, 730), radius=12, fill="white", outline="#DBE3EF", width=2)
        draw.text((170, 285), title, fill="#111827", font=h_font)
        draw.line((170, 358, 1328, 358), fill="#E2E8F0", width=2)
        for idx, line in enumerate(lines):
            draw.text((200, 418 + idx * 72), line, fill="#334155", font=body_font)
        draw.ellipse((1100, 420, 1240, 560), fill=color)
        draw.ellipse((970, 510, 1060, 600), fill=color)
        draw.ellipse((1220, 560, 1320, 660), fill=color)
        out = ASSET_DIR / filename
        img.save(out)
        paths.append(out)
    return paths


def build_pdf(image_paths: list[Path]) -> Path:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    pdfmetrics.registerFont(UnicodeCIDFont("STSong-Light"))

    styles = getSampleStyleSheet()
    title = ParagraphStyle("TitleCN", parent=styles["Title"], fontName="STSong-Light", fontSize=22, leading=28)
    h2 = ParagraphStyle("H2CN", parent=styles["Heading2"], fontName="STSong-Light", fontSize=15, leading=20, spaceBefore=10)
    body = ParagraphStyle("BodyCN", parent=styles["BodyText"], fontName="STSong-Light", fontSize=10.5, leading=17)
    code = ParagraphStyle(
        "CodeCN",
        parent=body,
        fontName="Courier",
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor("#111827"),
        backColor=colors.HexColor("#F1F5F9"),
        borderPadding=6,
    )

    doc = SimpleDocTemplate(str(PDF_PATH), pagesize=A4, rightMargin=1.7 * cm, leftMargin=1.7 * cm, topMargin=1.6 * cm, bottomMargin=1.6 * cm)
    story = [
        Paragraph("A3 语义表示与对比分析系统实验报告", title),
        Spacer(1, 0.3 * cm),
        Paragraph("项目路径：a3/；入口文件：a3/streamlit_app.py；部署方式：GitHub + Streamlit Community Cloud。", body),
        Paragraph("一、Vibe Coding Prompt 记录", h2),
        Paragraph("Prompt 1：使用 Python Streamlit 开发英文语义表示实验平台，包含 TF-IDF/LSA、Word2Vec、GloVe、FastText/Sent2Vec 四个模块，并共用同一份英文语料。", body),
        Paragraph("Prompt 2：第一个标签页按句子切分语料，计算 TF-IDF 矩阵，展示 Top 5 关键词，并用 TruncatedSVD 做 LSA 二维可视化。", body),
        Paragraph("Prompt 3：第二个标签页使用 gensim 训练 Word2Vec，单选切换 CBOW 与 Skip-Gram，滑动调整 window，输出查询词 Top 5 相似词。", body),
        Paragraph("Prompt 4：第三个标签页使用 gensim.downloader 加载 glove-twitter-25，实现 A - B + C 类比和双词相似度。", body),
        Paragraph("Prompt 5：第四个标签页训练 FastText，比较 Word2Vec OOV KeyError 与 FastText 子词向量，并用平均池化实现简单 Sent2Vec。", body),
        Paragraph("二、核心代码摘录", h2),
        Paragraph("TfidfVectorizer + TruncatedSVD；Word2Vec(sg=0/1)；keyed_vectors.most_similar(positive=[A, C], negative=[B])；FastText 子词向量；句向量 average pooling。", code),
        Paragraph("三、系统运行截图", h2),
        Paragraph("下列图片为报告配套界面图。实际提交前，可运行 streamlit run streamlit_app.py 后用浏览器截图替换。", body),
    ]

    for path in image_paths:
        story.append(Spacer(1, 0.2 * cm))
        story.append(Image(str(path), width=16.2 * cm, height=9.0 * cm))

    story.extend(
        [
            PageBreak(),
            Paragraph("四、实验观察与总结", h2),
            Paragraph("TF-IDF 能突出小语料中更有区分度的词；LSA 通过矩阵分解把词-文档矩阵压缩到二维空间，便于观察共现结构。", body),
            Paragraph("CBOW 与 Skip-Gram 的预测目标不同，因此在小语料实时训练时，同一个查询词的 Top 5 相似词可能发生变化。", body),
            Paragraph("GloVe 通过全局共现统计学习词向量，king - man + woman 等类比任务可以观察向量空间中的线性语义关系。", body),
            Paragraph("FastText 使用字符 n-gram，可以为 computeer 这样的拼写错误词生成向量；平均池化句向量虽然简单，但能快速估计句子整体相似度。", body),
            Spacer(1, 0.4 * cm),
            Table(
                [["提交文件", "说明"], ["streamlit_app.py", "Streamlit 四模块页面"], ["semantic_core.py", "核心算法与工具函数"], ["requirements.txt", "部署依赖"], ["A3_实验报告_可打印.html / PDF", "实验报告"]],
                colWidths=[5.2 * cm, 10.5 * cm],
                style=TableStyle(
                    [
                        ("FONTNAME", (0, 0), (-1, -1), "STSong-Light"),
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2FF")),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ]
                ),
            ),
        ]
    )
    doc.build(story)
    return PDF_PATH


if __name__ == "__main__":
    paths = make_demo_pngs()
    pdf = build_pdf(paths)
    print(f"Generated {pdf}")
