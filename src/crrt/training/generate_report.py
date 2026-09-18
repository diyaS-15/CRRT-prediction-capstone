import base64
import csv
import json
from datetime import datetime
from pathlib import Path

REPORTS_DIR = Path("reports")
OUT_HTML = REPORTS_DIR / "summary_report.html"

def read_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def read_csv_rows(path: Path, max_rows=50):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if len(rows) > max_rows + 1:
        rows = rows[: max_rows + 1]
    return rows

def img_to_data_uri(path: Path):
    if not path.exists():
        return None
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def html_table(rows):
    if not rows:
        return "<p><em>Missing</em></p>"
    header = rows[0]
    body = rows[1:]
    th = "".join(f"<th>{h}</th>" for h in header)
    trs = []
    for r in body:
        tds = "".join(f"<td>{c}</td>" for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return f"""
    <table>
      <thead><tr>{th}</tr></thead>
      <tbody>{"".join(trs)}</tbody>
    </table>
    """

def section(title, inner_html):
    return f"""
    <div class="card">
      <h2>{title}</h2>
      {inner_html}
    </div>
    """

def headline_metrics_card(model_name, metrics):
    """Lead with recall (sensitivity) and PR-AUC, not accuracy. With CRRT this
    imbalanced (~17% positive), a model that always predicts "no CRRT" scores
    ~83% accuracy while missing every true case — accuracy alone would
    overstate how good the model is at the thing that actually matters
    clinically: not missing a patient who needs CRRT. Recall and PR-AUC are
    the honest headline; accuracy is shown only as a secondary reference."""
    if not metrics:
        return "<p><em>Missing</em></p>"
    test_recall = metrics.get("test_sensitivity")
    test_pr_auc = metrics.get("test_pr_auc")
    test_precision = metrics.get("test_precision")
    test_accuracy = metrics.get("test_accuracy")
    test_fn = metrics.get("test_fn")
    test_tp = metrics.get("test_tp")

    def pct(v):
        return f"{v:.1%}" if isinstance(v, (int, float)) else "N/A"

    missed = (
        f"<p class=\"muted\">Missed {test_fn} of {test_fn + test_tp} true CRRT cases on the test set.</p>"
        if isinstance(test_fn, int) and isinstance(test_tp, int) and (test_fn + test_tp) > 0
        else ""
    )

    return f"""
    <div class="headline">
      <div class="stat stat-primary">
        <div class="stat-label">Test Recall (Sensitivity)</div>
        <div class="stat-value">{pct(test_recall)}</div>
        <div class="stat-sub">share of true CRRT cases caught</div>
      </div>
      <div class="stat stat-primary">
        <div class="stat-label">Test PR-AUC</div>
        <div class="stat-value">{pct(test_pr_auc)}</div>
        <div class="stat-sub">precision/recall tradeoff, imbalance-aware</div>
      </div>
      <div class="stat stat-secondary">
        <div class="stat-label">Test Precision</div>
        <div class="stat-value">{pct(test_precision)}</div>
      </div>
      <div class="stat stat-secondary">
        <div class="stat-label">Test Accuracy</div>
        <div class="stat-value">{pct(test_accuracy)}</div>
        <div class="stat-sub">reference only — misleading alone on imbalanced data</div>
      </div>
    </div>
    {missed}
    """

def main():
    REPORTS_DIR.mkdir(exist_ok=True)

    # Load files (XGB + CatBoost)
    xgb_metrics = read_json(REPORTS_DIR / "xgb_metrics.json")
    cat_metrics = read_json(REPORTS_DIR / "cat_metrics.json")

    xgb_cm = read_json(REPORTS_DIR / "confusion_matrix.json")
    cat_cm = read_json(REPORTS_DIR / "cat-confusion_matrix.json")

    xgb_split = read_json(REPORTS_DIR / "split_check.json")
    cat_split = read_json(REPORTS_DIR / "cat-split_check.json")

    xgb_importance = read_csv_rows(REPORTS_DIR / "xgb_feature_importance.csv")
    cat_importance = read_csv_rows(REPORTS_DIR / "cat_feature_importance.csv")

    shap_img_uri = img_to_data_uri(REPORTS_DIR / "shap_summary.png")

    xgb_fn = read_csv_rows(REPORTS_DIR / "test_false_negatives.csv")
    cat_fn = read_csv_rows(REPORTS_DIR / "cat-test_false_negatives.csv")

    def pretty_json(obj):
        if obj is None:
            return "<p><em>Missing</em></p>"
        return f"<pre>{json.dumps(obj, indent=2)}</pre>"

    # Build HTML
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>CRRT Model Summary Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #fafafa; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #666; margin-top: 0; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
    .card {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 16px; }}
    pre {{ background: #f3f3f3; padding: 12px; border-radius: 8px; overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; font-size: 14px; }}
    th {{ background: #f0f0f0; text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 8px; }}
    .headline {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }}
    .stat {{ flex: 1 1 120px; border-radius: 8px; padding: 10px 12px; }}
    .stat-primary {{ background: #eaf3ff; border: 1px solid #b6d7ff; }}
    .stat-secondary {{ background: #f3f3f3; border: 1px solid #ddd; }}
    .stat-label {{ font-size: 12px; color: #555; text-transform: uppercase; letter-spacing: 0.03em; }}
    .stat-value {{ font-size: 26px; font-weight: bold; }}
    .stat-sub {{ font-size: 11px; color: #777; }}
  </style>
</head>
<body>
  <h1>CRRT Prediction – Sponsor Summary Report</h1>
  <p class="muted">Generated: {now}</p>

  <div class="card">
    <h2>Why recall, not accuracy, is the headline</h2>
    <p>CRRT is rare in this data (~17% of patients). A model that always predicts
    "no CRRT" would score ~83% accuracy while catching zero true cases — accuracy
    alone would make a useless model look strong. The numbers that actually matter
    for a clinical screening tool are <strong>recall/sensitivity</strong> (of the
    patients who truly needed CRRT, how many did we flag?) and
    <strong>PR-AUC</strong> (precision/recall tradeoff, which — unlike ROC-AUC —
    isn't inflated by the large majority-class). Accuracy is reported below for
    completeness but should not be read as the primary success metric.</p>
  </div>

  <div class="grid">
    {section("Headline Metrics (XGBoost)", headline_metrics_card("XGBoost", xgb_metrics))}
    {section("Headline Metrics (CatBoost)", headline_metrics_card("CatBoost", cat_metrics))}
  </div>

  {section("Dataset / Split Check (XGBoost)", pretty_json(xgb_split))}
  {section("Dataset / Split Check (CatBoost)", pretty_json(cat_split))}

  <div class="grid">
    {section("Full Metrics (XGBoost)", pretty_json(xgb_metrics))}
    {section("Full Metrics (CatBoost)", pretty_json(cat_metrics))}
  </div>

  <div class="grid">
    {section("Confusion Matrix (XGBoost)", pretty_json(xgb_cm))}
    {section("Confusion Matrix (CatBoost)", pretty_json(cat_cm))}
  </div>

  <div class="grid">
    {section("Feature Importance (XGBoost) – top rows", html_table(xgb_importance))}
    {section("Feature Importance (CatBoost) – top rows", html_table(cat_importance))}
  </div>

  {section("SHAP Summary Plot", '<p><em>Missing shap_summary.png</em></p>' if not shap_img_uri else f'<img src="{shap_img_uri}" />')}

  <div class="grid">
    {section("Test False Negatives (XGBoost)", html_table(xgb_fn))}
    {section("Test False Negatives (CatBoost)", html_table(cat_fn))}
  </div>

  <div class="card">
    <h2>Notes / Interpretation (fill in)</h2>
    <ul>
      <li>What the model is predicting (CRRT need within the defined time window).</li>
      <li>How to interpret sensitivity/recall (catching true CRRT cases).</li>
      <li>Main limitations: synthetic data, generalization, threshold selection.</li>
    </ul>
  </div>
</body>
</html>
"""

    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"✅ Wrote: {OUT_HTML}")

if __name__ == "__main__":
    main()