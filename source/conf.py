# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GPGPU Architectures and Simulations'
copyright = 'Copyright © 2024 MASA-Laboratory <masa-lab@outlook.com>'
author = "Jianchao Yang"
release = 'v1.0.0'
affilication = "MASA-Laboratory, NUDT"
email = "masa-lab@outlook.com"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# It must use `pip install urllib3==1.26.6`
extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    "myst_parser",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.imgmath"
]

intersphinx_disabled_domains = ['std']

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_show_sourcelink = False
numfig = True

# -- Options for EPUB output
epub_show_urls = 'footnote'

# -- Options for LaTex output
# "atendofbody": r"\vspace*{\fill}\begin{flushright}\textbf{Jianchao Yang}\\\textbf{MASA-Laboratory, NUDT}\\\textbf{masa-lab@outlook.com}\end{flushright}",
    
latex_elements = {
    "papersize": "a4paper",
    'fncychap': r'\usepackage[Bjornstrup]{fncychap}',
    'preamble': r'''
\usepackage{ctex}
\usepackage{bm}
\usepackage{fontspec}
\newcommand{\myproject}{''' + project + r'''}
\newcommand{\mycopyright}{''' + copyright + r'''}
\newcommand{\myauthor}{''' + author + r'''}
\newcommand{\myrelease}{''' + release + r'''}
\newcommand{\myaffilication}{''' + affilication + r'''}
\newcommand{\myemail}{''' + email + r'''}
\newcommand{\mycustomcontent}{
    \newpage
    \thispagestyle{empty}
    \begin{center}
        \vspace*{\fill}
        \textbf{\Large{\bf{\myproject}}}\\[5pt]
        \textbf{Version: \myrelease}\\[5pt]
        \textit{\mycopyright}\\[5pt]
        \textit{Created by: \myauthor}\\[5pt]
        \textit{Affiliation: \myaffilication}\\[5pt]
        \textit{Email: \myemail}\\[5pt]
        \textit{Last updated: \today}\\[5pt]
        \textit{All rights reserved.}\\
        \vspace*{\fill}
    \end{center}
    \newpage
}
''',
    'tableofcontents': r'''
\mycustomcontent
\tableofcontents
''',
    'classoptions': ',oneside',

}