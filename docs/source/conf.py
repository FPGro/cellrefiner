import os
import sys
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../cellrefiner'))

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'cellrefiner'
copyright = '2025, Eric Bourgain-Chang, Xiangyu Kuang'
author = 'Eric Bourgain-Chang, Xiangyu Kuang'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_mdinclude',
    'sphinx.ext.autodoc',        # 自动生成API文档
    'sphinx.ext.viewcode',       # 源代码链接
    'sphinx.ext.napoleon',       # Google/NumPy风格docstring支持
    'sphinx.ext.autosummary',    # 自动摘要
    'sphinx_autodoc_typehints',  # 类型提示支持
    'sphinx.ext.githubpages',    # GitHub Pages支持
    'nbsphinx',
]

# Generate API documentation
autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
