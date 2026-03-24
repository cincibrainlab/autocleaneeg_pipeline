import importlib


def test_visualization_package_imports_without_circular_dependency():
    module = importlib.import_module("autoclean.functions.visualization")
    assert module is not None


def test_mixins_viz_ica_imports_after_visualization_package():
    importlib.import_module("autoclean.functions.visualization")
    module = importlib.import_module("autoclean.mixins.viz.ica")
    assert hasattr(module, "ICAReportingMixin")
