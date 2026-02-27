"""
build_pdf.py
Converts paper.md → paper.pdf
Embeds all actual chart images from ids_results and ids_results/shap_outputs
"""

import os, base64, re
import markdown
from xhtml2pdf import pisa

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE        = r'D:\base\mullti'
RESULTS     = os.path.join(BASE, 'ids_results')
SHAP_OUT    = os.path.join(RESULTS, 'shap_outputs')
PAPER_MD    = os.path.join(BASE, 'paper.md')
PAPER_PDF   = os.path.join(BASE, 'paper.pdf')

# ── Helper: embed image as base64 data-URI ────────────────────────────────────
def img_tag(path, caption, width='520px'):
    if not os.path.exists(path):
        return f'<p style="color:red">[Missing: {path}]</p>'
    with open(path, 'rb') as f:
        data = base64.b64encode(f.read()).decode()
    return (
        f'<div class="figure">'
        f'<img src="data:image/png;base64,{data}" style="width:{width}"/>'
        f'<p class="caption">{caption}</p>'
        f'</div>'
    )

# ── Build figure blocks ───────────────────────────────────────────────────────
fig1_placeholder = (
    '<p><em>Fig. 1: Transformer IDS Architecture</em></p>'
)
fig1_html = '<div class="figure"><p class="caption"><strong>Fig. 1:</strong> Transformer IDS Architecture — Linear Embedding → Positional Encoding → Layer Norm → 2× Transformer Encoder Layer → Classification Head (82,786 total parameters)</p></div>'

fig2_html = img_tag(
    os.path.join(RESULTS, 'training_dashboard.png'),
    '<strong>Fig. 2:</strong> Training Dashboard — (a) Training Loss Curve, (b) Validation Accuracy Curve, (c) Epoch Time per Epoch, (d) Learning Rate Schedule over 50 epochs'
)

fig3_html = img_tag(
    os.path.join(RESULTS, 'per_class_metrics.png'),
    '<strong>Fig. 3:</strong> Per-Class Precision, Recall, and F1-Score for all 34 traffic classes'
)

fig4_html = img_tag(
    os.path.join(RESULTS, 'confusion_matrix.png'),
    '<strong>Fig. 4:</strong> Confusion Matrix (raw counts) — Test Set (72,743 samples)'
)

fig5_html = img_tag(
    os.path.join(RESULTS, 'confusion_matrix_normalized.png'),
    '<strong>Fig. 5:</strong> Row-Normalised Confusion Matrix — each cell shows fraction of true-class samples predicted as each class'
)

fig6_html = img_tag(
    os.path.join(SHAP_OUT, 'shap_bar_summary.png'),
    '<strong>Fig. 6:</strong> SHAP Global Feature Importance — Mean |SHAP value| per feature across all 34 classes'
)

fig7_html = img_tag(
    os.path.join(SHAP_OUT, 'shap_global_importance.png'),
    '<strong>Fig. 7:</strong> SHAP Global Importance Bar Chart — Top features ranked by global mean absolute SHAP value'
)

fig8_html = img_tag(
    os.path.join(RESULTS, 'feature_importance.png'),
    '<strong>Fig. 8:</strong> Gradient-Based Feature Importance (input-space attribution integrated into model inference path)'
)

fig9_html = img_tag(
    os.path.join(RESULTS, 'timing_summary.png'),
    '<strong>Fig. 9:</strong> Pipeline Stage Timing Summary — Training dominates at 97.7% of 43.3 min total'
)

# ── Select representative SHAP beeswarm/waterfall figures ─────────────────────
shap_beeswarm_classes = [
    'DDoS-ICMP_Flood', 'DDoS-SYN_Flood', 'BenignTraffic', 'Mirai-udpplain', 'MITM-ArpSpoofing'
]
shap_beeswarm_html = '<h3>SHAP Per-Class Beeswarm Summary Plots (Representative Classes)</h3>'
for cls in shap_beeswarm_classes:
    fname = f'shap_summary_{cls}.png'
    shap_beeswarm_html += img_tag(
        os.path.join(SHAP_OUT, fname),
        f'SHAP Beeswarm — Class: {cls}',
        width='460px'
    )

waterfall_html = '<h3>SHAP Waterfall Plots (Individual Sample Explanations)</h3>'
for i in range(1, 6):
    waterfall_html += img_tag(
        os.path.join(SHAP_OUT, f'shap_waterfall_sample{i}.png'),
        f'SHAP Waterfall — Sample {i}',
        width='460px'
    )

# ── Read and convert Markdown → HTML ─────────────────────────────────────────
with open(PAPER_MD, 'r', encoding='utf-8') as f:
    md_text = f.read()

body_html = markdown.markdown(
    md_text,
    extensions=['tables', 'fenced_code', 'nl2br', 'toc']
)

# ── Inject real figure blocks in place of *Fig. X* placeholders ──────────────
replacements = [
    # replace the architecture placeholder paragraph
    (r'<p><em>Fig\. 1: Transformer IDS Architecture</em></p>',          fig1_html),
    (r'<p><em>Fig\. 2:.*?Training Dashboard.*?</em></p>',               fig2_html),
    (r'<p><em>Fig\. 3: Per-Class.*?</em></p>',                          fig3_html),
    (r'<p><em>Fig\. 4: Confusion Matrix \(raw counts\)</em></p>',       fig4_html),
    (r'<p><em>Fig\. 5: Normalised Confusion Matrix</em></p>',           fig5_html),
    (r'<p><em>Fig\. 6: SHAP Global.*?</em></p>',                        fig6_html),
    (r'<p><em>Fig\. 7: SHAP Beeswarm.*?</em></p>',                      fig7_html),
]

for pattern, replacement in replacements:
    body_html = re.sub(pattern, replacement, body_html, flags=re.DOTALL)

# ── Append additional figures section ─────────────────────────────────────────
appendix = f"""
<hr/>
<h2>Appendix A — Additional Experimental Figures</h2>

{fig8_html}
{fig9_html}

<h2>Appendix B — SHAP Explainability Figures</h2>
{shap_beeswarm_html}
{waterfall_html}
"""

# ── Full HTML document with IEEE-inspired CSS ─────────────────────────────────
CSS = """
@page {
    size: A4;
    margin: 2.2cm 2cm 2.2cm 2cm;
}
body {
    font-family: "Times New Roman", Times, serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #000;
}
h1 {
    font-size: 15pt;
    text-align: center;
    margin-bottom: 4pt;
    font-weight: bold;
}
h2 {
    font-size: 11pt;
    font-variant: small-caps;
    border-bottom: 1px solid #333;
    margin-top: 14pt;
    margin-bottom: 4pt;
    padding-bottom: 2pt;
}
h3 {
    font-size: 10pt;
    font-style: italic;
    font-weight: bold;
    margin-top: 8pt;
}
h4 { font-size: 10pt; font-weight: bold; margin-top: 6pt; }
p   { margin: 4pt 0; text-align: justify; }
ul, ol { margin: 4pt 0 4pt 18pt; }
li  { margin-bottom: 2pt; }
code, pre {
    font-family: "Courier New", monospace;
    font-size: 8pt;
    background: #f4f4f4;
    border: 1px solid #ddd;
    padding: 2pt 4pt;
    border-radius: 2pt;
}
pre { padding: 6pt; display: block; white-space: pre-wrap; }

/* Tables */
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
    margin: 8pt 0;
    page-break-inside: avoid;
}
thead tr { background-color: #1a3a5c; color: white; }
thead th { padding: 4pt 5pt; text-align: left; font-weight: bold; }
tbody tr:nth-child(even) { background-color: #f0f4f8; }
tbody td { padding: 3pt 5pt; border-bottom: 1px solid #ccc; vertical-align: top; }

/* Figures */
.figure {
    margin: 10pt auto;
    text-align: center;
    page-break-inside: avoid;
}
.figure img {
    display: block;
    margin: 0 auto;
    max-width: 100%;
    border: 1px solid #ccc;
    padding: 3pt;
}
.caption {
    font-size: 8.5pt;
    font-style: italic;
    color: #333;
    margin-top: 4pt;
    text-align: center;
}

/* Abstract block */
blockquote {
    border-left: 3px solid #1a3a5c;
    margin: 8pt 0;
    padding: 6pt 12pt;
    background: #f8faff;
    font-style: italic;
    font-size: 9.5pt;
}
hr {
    border: none;
    border-top: 1px solid #999;
    margin: 14pt 0;
}
strong { font-weight: bold; }
em     { font-style: italic; }
"""

full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<style>{CSS}</style>
</head>
<body>
{body_html}
{appendix}
</body>
</html>"""

# ── Save HTML for debugging ────────────────────────────────────────────────────
html_path = os.path.join(BASE, 'paper_debug.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(full_html)
print(f"HTML saved → {html_path}")

# ── Convert to PDF ────────────────────────────────────────────────────────────
print("Converting to PDF …")
with open(PAPER_PDF, 'wb') as pdf_file:
    result = pisa.CreatePDF(
        full_html.encode('utf-8'),
        dest=pdf_file,
        encoding='utf-8'
    )

if result.err:
    print(f"PDF generation errors: {result.err}")
else:
    size_mb = os.path.getsize(PAPER_PDF) / 1e6
    print(f"PDF saved → {PAPER_PDF}  ({size_mb:.1f} MB)")
    print("Done!")
