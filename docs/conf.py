"""Configuration file for the Sphinx documentation builder.

This file contains settings for Sphinx to build the Autoclean documentation.
"""

import os
import sys
from datetime import date

# Add the exact path to the autoclean package
sys.path.insert(0, os.path.abspath('../../src'))
# Path to the root directory
sys.path.insert(0, os.path.abspath('../..'))

# Import the module to verify it's available
try:
    import autoclean
    print(f"Autoclean module found at: {autoclean.__file__}")
except ImportError as e:
    print(f"WARNING: Unable to import autoclean: {e}")

# -- Project information -----------------------------------------------------
project = 'Autoclean'
copyright = f'2024-{date.today().year}, Cincibrainlab Team'
author = 'Gavin Gammoh, Ernest Pedapati'

# The full version, including alpha/beta/rc tags
# from autoclean import __version__ as version
version = '2.1.0'  # Hardcoded to avoid import errors
release = version

# -- General configuration ---------------------------------------------------
extensions = [
    # Sphinx core extensions
    'sphinx.ext.autodoc',         # Include documentation from docstrings
    'sphinx.ext.autosummary',     # Generate autodoc summaries
    'sphinx.ext.doctest',         # Test snippets in the documentation
    'sphinx.ext.intersphinx',     # Link to other projects' documentation
    'sphinx.ext.todo',            # Support for todo items
    'sphinx.ext.coverage',        # Collect doc coverage stats
    'sphinx.ext.mathjax',         # Render math via JavaScript
    'sphinx.ext.ifconfig',        # Conditional content based on config values
    'sphinx.ext.viewcode',        # Add links to view source code
    'sphinx.ext.githubpages',     # GitHub pages support
    
    # External extensions
    'numpydoc',                   # Support for NumPy style docstrings
    'sphinx_gallery.gen_gallery', # Generate gallery of examples
]

# Configure sphinx-gallery
sphinx_gallery_conf = {
    'examples_dirs': [],  # No example directories
    'gallery_dirs': [],   # No gallery directories
    'backreferences_dir': None,
    'doc_module': ('autoclean',),
    'reference_url': {
        'autoclean': None,
    },
}

# Configure autodoc
autodoc_default_options = {
    'members': True,
    'undoc-members': False,
    'show-inheritance': False,
    'member-order': 'bysource',
    'exclude-members': 'VALUES, DEBUG, INFO, SUCCESS, HEADER, WARNING, ERROR, CRITICAL',
}

# Hide type hints in signatures
autodoc_typehints = 'none'
autodoc_typehints_format = 'short'

# Configure autosummary
autosummary_generate = True
autosummary_imported_members = False  # Disable imported members to avoid missing function errors

# Configure NumPy docstrings
numpydoc_show_class_members = False
numpydoc_class_members_toctree = True
numpydoc_xref_param_type = True
numpydoc_xref_aliases = {
    'ndarray': 'numpy.ndarray',
    'DataFrame': 'pandas.DataFrame',
    'Path': 'pathlib.Path',
}
numpydoc_show_inherited_class_members = False
numpydoc_attributes_as_param_list = False
numpydoc_use_blockquotes = False
numpydoc_use_plots = False
numpydoc_validate = False  # Disable validation for now to allow the build to complete

# Improve list rendering
numpydoc_xref_ignore = {'optional', 'type', 'of'}

# Remove conflicting validation checks when validation is disabled
# numpydoc_validation_checks = {"all"}

# Configure intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'mne': ('https://mne.tools/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'pyprep': ('https://pyprep.readthedocs.io/en/latest/', None),
}

# Add any paths that contain templates here
templates_path = ['_templates']

# List of patterns to exclude from source
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# -- Options for HTML output -------------------------------------------------
html_theme = 'pydata_sphinx_theme'  # Use the PyData Sphinx theme like MNE
html_theme_options = {
    'icon_links': [
        {
            'name': 'GitHub',
            'url': 'https://github.com/cincibrainlab/autoclean_complete',
            'icon': 'fab fa-github-square',
            'type': 'fontawesome',
        },
    ],
    'logo': {
        'text': 'AutoClean',
    },
    'navigation_with_keys': True,
    'show_prev_next': False,
    'navbar_end': ['navbar-icon-links'],
    'navigation_depth': 4,
    'collapse_navigation': False,
    'show_nav_level': 2,
    'primary_sidebar_end': ["indices.html"],
    'secondary_sidebar_items': ["page-toc"],
    'navbar_align': "content",
    'globaltoc_collapse': True,
    'globaltoc_includehidden': True,
    'globaltoc_maxdepth': 3,
}

# Add any paths that contain custom static files
html_static_path = ['_static']  # Updated to include custom CSS

# Add custom CSS files
html_css_files = [
    'css/custom.css',
]

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '11pt',
}

# -- Extension configuration -------------------------------------------------
# Configure todo extension
todo_include_todos = True

# Mock imports for autodoc
autodoc_mock_imports = [
    'numpy', 'pandas', 'matplotlib', 'scipy', 'mne', 
    'rich', 'dotenv', 'pyyaml', 'schema', 'mne_bids', 
    'pybv', 'torch', 'pyprep', 'eeglabio', 'autoreject', 'ulid',
    'unqlite', 'loguru', 'reportlab', 'pyqt5', 'pyvistaqt',
    'tqdm', 'yaml', 'asyncio', 'json', 'datetime', 'xarray',
    'mne_icalabel', 'cython', 'pydantic', 'nibabel', 'platformdirs',
    'defusedxml', 'python_ulid', 'networkx', 'bctpy', 'fooof',
    'mne_connectivity', 'fastparquet', 'python_picard', 'picard',
] 