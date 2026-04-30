"""Sphinx configuration for the MicroI2I documentation site."""

from __future__ import annotations

import datetime as _dt
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = Path(__file__).resolve().parent

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

project = "MicroI2I Documentation"
author = "MicroI2I contributors"
copyright = f"{_dt.datetime.now().year}, {author}"

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "attrs_block",
    "attrs_inline",
    "substitution",
]
myst_heading_anchors = 3
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2
templates_path = ["_templates"]
exclude_patterns = ["_build", ".DS_Store", "**/__pycache__/**"]
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
master_doc = "index"
language = "en"
pygments_style = "sphinx"
html_theme = "furo"
html_title = "MicroI2I Documentation"
html_baseurl = os.environ.get("MICROI2I_DOCS_BASEURL", "")
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_show_sourcelink = True
html_copy_source = False
html_last_updated_fmt = "%Y-%m-%d"
html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "top_of_page_button": "edit",
    "light_css_variables": {
        "color-brand-primary": "#176b87",
        "color-brand-content": "#124e63",
        "color-api-name": "#16324f",
        "color-api-pre-name": "#16324f",
        "color-sidebar-background": "#102a43",
        "color-sidebar-background-border": "#081826",
        "color-sidebar-item-background": "#102a43",
        "color-sidebar-item-background--current": "#1b4965",
        "color-sidebar-item-background--hover": "#1d5b79",
        "color-sidebar-item-expander-background": "#1e6091",
        "color-sidebar-item-expander-background--hover": "#247ba0",
        "color-sidebar-link-text": "#edf6f9",
        "color-sidebar-link-text--top-level": "#ffe8a3",
        "color-sidebar-caption-text": "#b8d8e8",
        "color-sidebar-brand-text": "#ffffff",
        "color-sidebar-search-background": "#0b1f33",
        "color-sidebar-search-background--focus": "#12385a",
        "color-sidebar-search-border": "#386f8f",
        "color-sidebar-search-foreground": "#f5fbff",
        "color-sidebar-search-icon": "#cce3ed",
    },
}

mathjax_local_path = DOCS / "_static" / "mathjax" / "es5" / "tex-mml-chtml.js"

if mathjax_local_path.exists():
    mathjax_path = "mathjax/es5/tex-mml-chtml.js"
else:
    mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@4/es5/tex-mml-chtml.js"

mathjax_common_config = {
    "tex": {
        "inlineMath": [["\\(", "\\)"]],
        "displayMath": [["\\[", "\\]"]],
        "packages": {"[+]": ["ams"]},
    }
}
mathjax4_config = mathjax_common_config
mathjax3_config = mathjax_common_config

nitpicky = False
numfig = True
