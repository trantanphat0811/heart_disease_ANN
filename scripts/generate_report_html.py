"""Generate a static HTML page that embeds EDA images, summary table and model metrics.

Usage:
    python scripts/generate_report_html.py

Outputs:
    web/index.html
"""
import base64
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"
OUT_DIR = ROOT / "web"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def embed_image(img_path: Path) -> str:
    with img_path.open("rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def read_metrics(metrics_path: Path) -> str:
    if not metrics_path.exists():
        return "No metrics file found. Run training first."
    return metrics_path.read_text()


def generate_html():
    # after splitting reports, CSVs are stored in reports/tables and images in reports/images
    summary_csv = REPORTS / "tables" / "summary_statistics.csv"
    metrics_txt = MODELS / "metrics.txt"

    if summary_csv.exists():
        df_summary = pd.read_csv(summary_csv, index_col=0)
        # render only first 10 rows/cols for readability
        html_table = df_summary.iloc[:, :10].to_html(classes="table table-striped", border=0)
    else:
        html_table = "<p>No summary_statistics.csv found.</p>"

    metrics = read_metrics(metrics_txt)

    # collect pngs from reports/images
    imgs_dir = REPORTS / "images"
    if imgs_dir.exists():
        imgs = list(imgs_dir.glob("*.png"))
    else:
        # fallback: look at reports root (backwards compatibility)
        imgs = list(REPORTS.glob("*.png"))

    img_tags = []
    for p in imgs:
        src = embed_image(p)
        img_tags.append((p.name, src))

    html = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Heart Disease Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.0/font/bootstrap-icons.css" rel="stylesheet">
    <style>
      * {{
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }}
      
      body {{
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        min-height: 100vh;
        padding: 40px 20px;
      }}
      
      .container {{
        max-width: 1200px;
      }}
      
      .header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 50px 30px;
        border-radius: 15px;
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
      }}
      
      .header h1 {{
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 15px;
      }}
      
      .header p {{
        font-size: 1.1rem;
        opacity: 0.95;
        margin: 0;
      }}
      
      .section {{
        background: white;
        border-radius: 12px;
        padding: 30px;
        margin-bottom: 30px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }}
      
      .section:hover {{
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
      }}
      
      .section h2 {{
        color: #667eea;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 20px;
        padding-bottom: 15px;
        border-bottom: 3px solid #667eea;
      }}
      
      .metrics-box {{
        background: linear-gradient(135deg, #f5f7fa 0%, #e9ecef 100%);
        border-left: 5px solid #667eea;
        padding: 20px;
        border-radius: 8px;
        font-family: 'Courier New', monospace;
        font-size: 0.95rem;
        line-height: 1.6;
        overflow-x: auto;
        color: #2d3436;
      }}
      
      .summary-table {{
        margin-top: 15px;
      }}
      
      .summary-table table {{
        background: white;
        border-collapse: collapse;
      }}
      
      .summary-table th {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 12px;
        text-align: left;
        font-weight: 600;
      }}
      
      .summary-table td {{
        padding: 10px 12px;
        border-bottom: 1px solid #e9ecef;
      }}
      
      .summary-table tbody tr:hover {{
        background-color: #f8f9ff;
      }}
      
      .plots-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
        gap: 25px;
        margin-top: 20px;
      }}
      
      .img-card {{
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
      }}
      
      .img-card:hover {{
        transform: translateY(-8px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
      }}
      
      .img-card-header {{
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        font-weight: 600;
        font-size: 1rem;
      }}
      
      .img-card-body {{
        padding: 15px;
      }}
      
      .img-card img {{
        max-width: 100%;
        height: auto;
        display: block;
        border-radius: 8px;
      }}
      
      footer {{
        text-align: center;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 2px solid #e9ecef;
        color: #636e72;
        font-size: 0.95rem;
      }}
      
      @media (max-width: 768px) {{
        .header h1 {{
          font-size: 1.8rem;
        }}
        
        .plots-grid {{
          grid-template-columns: 1fr;
        }}
        
        .section {{
          padding: 20px;
        }}
      }}
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1><i class="bi bi-heart-pulse"></i> Heart Disease Report</h1>
        <p>Comprehensive EDA, model metrics, and visualization dashboard</p>
        <div style="margin-top: 20px; display: flex; justify-content: center; gap: 20px;">
          <a href="index.html" style="color: white; text-decoration: none; padding: 8px 16px; background: rgba(255,255,255,0.2); border-radius: 6px; font-size: 0.95rem;"><i class="bi bi-graph-up"></i> Report</a>
          <a href="train.html" style="color: white; text-decoration: none; padding: 8px 16px; background: rgba(255,255,255,0.2); border-radius: 6px; font-size: 0.95rem;"><i class="bi bi-cpu"></i> Train Model</a>
        </div>
      </div>

      <div class="section">
        <h2><i class="bi bi-graph-up"></i> Model Performance Metrics</h2>
        <div class="metrics-box">{metrics}</div>
      </div>

      <div class="section">
        <h2><i class="bi bi-table"></i> Summary Statistics</h2>
        <div class="summary-table">{html_table}</div>
      </div>

      <div class="section">
        <h2><i class="bi bi-bar-chart"></i> Data Visualizations</h2>
        <div class="plots-grid">
"""

    # add images
    for name, src in img_tags:
        # extract friendly name from filename
        friendly_name = name.replace('.png', '').replace('_', ' ').title()
        html += f"""
          <div class="img-card">
            <div class="img-card-header">{friendly_name}</div>
            <div class="img-card-body">
              <img src="{src}" alt="{name}" />
            </div>
          </div>
"""

    html += """
        </div>
      </div>
      
      <footer>
        <p><i class="bi bi-info-circle"></i> Auto-generated report | Last updated: 16 Nov 2025</p>
        <p style="font-size: 0.85rem; color: #95a5a6;">Generated by <code>scripts/generate_report_html.py</code></p>
      </footer>
    </div>
  </body>
</html>
"""

    out_path = OUT_DIR / "index.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    generate_html()
