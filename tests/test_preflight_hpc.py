from __future__ import annotations

from types import SimpleNamespace

from scripts.preflight_hpc import (
    CRITICAL_IMPORT_MODULES,
    Reporter,
    build_parser,
    check_imports,
)


class _FakeTransformers:
    def __init__(self, *, gemma_error: Exception | None = None) -> None:
        self.accessed: list[str] = []
        self.gemma_error = gemma_error

    def __getattr__(self, name: str):
        self.accessed.append(name)
        if name == "Gemma3Config":
            return object()
        if name == "Gemma3ForConditionalGeneration":
            if self.gemma_error is not None:
                raise self.gemma_error
            return object()
        raise AttributeError(name)


def test_check_imports_loads_every_module_and_forces_gemma_lazy_import() -> None:
    attempted: list[str] = []
    transformers = _FakeTransformers()

    def importer(name: str):
        attempted.append(name)
        return transformers if name == "transformers" else SimpleNamespace()

    reporter = Reporter()
    check_imports(reporter, importer=importer)

    assert attempted == list(CRITICAL_IMPORT_MODULES)
    assert transformers.accessed == [
        "Gemma3Config",
        "Gemma3ForConditionalGeneration",
    ]
    assert reporter.failures == 0


def test_check_imports_reports_abi_failure_and_continues(capsys) -> None:
    attempted: list[str] = []
    transformers = _FakeTransformers()

    def importer(name: str):
        attempted.append(name)
        if name == "datasets":
            raise ImportError("libstdc++.so.6: version `GLIBCXX_3.4.31' not found")
        return transformers if name == "transformers" else SimpleNamespace()

    reporter = Reporter()
    check_imports(reporter, importer=importer)

    output = capsys.readouterr().out
    assert attempted == list(CRITICAL_IMPORT_MODULES)
    assert reporter.failures == 1
    assert "Runtime import failed for datasets" in output
    assert "GLIBCXX_3.4.31" in output
    assert "Runtime import succeeded: fire" in output


def test_check_imports_reports_gemma_lazy_import_failure(capsys) -> None:
    transformers = _FakeTransformers(
        gemma_error=ImportError("pyarrow failed to load: GLIBCXX_3.4.31")
    )

    def importer(name: str):
        return transformers if name == "transformers" else SimpleNamespace()

    reporter = Reporter()
    check_imports(reporter, importer=importer)

    output = capsys.readouterr().out
    assert reporter.failures == 1
    assert "Gemma 3 multimodal lazy import failed" in output
    assert "GLIBCXX_3.4.31" in output


def test_skip_import_check_flag() -> None:
    assert build_parser().parse_args(["--skip-import-check"]).skip_import_check
