"""Génération de rapports HTML à partir des résultats du pipeline.

Le module assemble figures, statistiques et métriques dans un template
Jinja2 pour produire une synthèse exploitable.
"""

import os

import jinja2


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
  """Construit un rapport HTML minimal en injectant les chemins utiles au template."""
    os.makedirs(out_dir, exist_ok=True)
  html = jinja2.Template(TEMPLATE).render(
    title=title,
    xai_root=xai_root,
    stats_root=stats_root,
    fig_root=fig_root,
  )
  with open(os.path.join(out_dir, "report.html"), "w", encoding="utf-8") as f:
    f.write(html)
