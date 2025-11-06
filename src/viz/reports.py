# src/viz/reports.py
import os, jinja2, json
TEMPLATE = """
<h1>{{ title }}</h1>
<h2>Résumé</h2>
<ul>
  <li>Figures: {{ fig_root }}</li>
  <li>Stats: {{ stats_root }}</li>
  <li>XAI: {{ xai_root }}</li>
</ul>
<h2>Commentaires</h2>
<p>Rapport généré automatiquement.</p>
"""
def build_report(title, xai_root, stats_root, fig_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    html = jinja2.Template(TEMPLATE).render(title=title, xai_root=xai_root, stats_root=stats_root, fig_root=fig_root)
    with open(os.path.join(out_dir, "report.html"), "w") as f: f.write(html)
