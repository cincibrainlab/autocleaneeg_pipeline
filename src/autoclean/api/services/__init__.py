"""Service modules used by the Serve/API layer.

Unlike ``autoclean.api.routes``, modules here contain no FastAPI
route decorators -- they are plain, importable Python that routes call
into. This keeps heavier logic (e.g. ICA refitting) testable without
spinning up the FastAPI app.
"""
