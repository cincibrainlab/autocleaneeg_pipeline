"""End-to-end MATLAB Engine integration coverage for AutoClean wrappers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest
def _python_version_dir(venv_dir: Path) -> str:
    site_packages_root = venv_dir / "lib"
    matches = sorted(path.name for path in site_packages_root.glob("python3.*"))
    if not matches:
        raise RuntimeError(f"No python3.* runtime found under {site_packages_root}")
    return matches[0]


def test_matlab_engine_runs_autoclean_wrappers(tmp_path: Path) -> None:
    """Exercise the real MATLAB engine through the AutoClean wrapper surface."""
    if os.environ.get("AUTOCLEAN_RUN_MATLAB_TESTS") != "1":
        pytest.skip("Set AUTOCLEAN_RUN_MATLAB_TESTS=1 to run real MATLAB integration tests.")

    repo_root = Path(__file__).resolve().parents[2]
    venv_dir = repo_root / ".venv"
    python_exe = venv_dir / "bin" / "python"

    if not venv_dir.exists():
        pytest.skip(f"Expected MATLAB-enabled .venv at {venv_dir}")
    if not python_exe.exists():
        pytest.skip(f"Expected Python executable at {python_exe}")

    python_version_dir = _python_version_dir(venv_dir)
    site_packages = venv_dir / "lib" / python_version_dir / "site-packages"
    if not (site_packages / "matlab").exists():
        pytest.skip("matlab package is not installed directly in .venv site-packages")

    matlab_script = tmp_path / "write_probe.m"
    output_json = tmp_path / "matlab_wrapper_probe.json"
    matlab_script.write_text(
        "\n".join(
            [
                "function write_probe(output_json)",
                "payload.sqrt_via_script = sqrt(49);",
                "payload.marker = 'matlab-script-ok';",
                "payload.output_path = char(output_json);",
                "json_text = jsonencode(payload);",
                "fid = fopen(char(output_json), 'w');",
                "fprintf(fid, '%s', json_text);",
                "fclose(fid);",
                "end",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    driver = tmp_path / "mwpython_driver.py"
    result_json = tmp_path / "driver_result.json"
    driver.write_text(
        "\n".join(
                [
                    "import json",
                    "import importlib.util",
                    "from pathlib import Path",
                    "",
                    f"module_path = Path({str((repo_root / 'src' / 'autoclean' / 'functions' / 'matlab.py'))!r})",
                    "spec = importlib.util.spec_from_file_location('autoclean_functions_matlab', module_path)",
                    "module = importlib.util.module_from_spec(spec)",
                    "assert spec is not None and spec.loader is not None",
                    "spec.loader.exec_module(module)",
                    "call_matlab = module.call_matlab",
                    "execute_matlab_config = module.execute_matlab_config",
                    f"output_json = Path({str(output_json)!r})",
                    f"result_json = Path({str(result_json)!r})",
                "sqrt_result = call_matlab('sqrt', 16.0, startup_options='')",
                    "execute_matlab_config(",
                    "    {",
                    "        'enabled': True,",
                    "        'value': {",
                    "            'kind': 'function',",
                    "            'entrypoint': 'write_probe',",
                    f"            'args': [{str(output_json)!r}],",
                    "            'paths': ['.'],",
                    "            'startup_options': '',",
                    "            'startup_timeout_seconds': 60.0,",
                    "            'nargout': 0,",
                    "        },",
                    "    },",
                    f"    base_path={str(tmp_path)!r},",
                    ")",
                "result_json.write_text(",
                "    json.dumps({",
                "        'sqrt_result': sqrt_result,",
                "        'script_output_exists': output_json.exists(),",
                "        'script_payload': json.loads(output_json.read_text(encoding=\"utf-8\")),",
                "    }),",
                "    encoding='utf-8',",
                ")",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(venv_dir)
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(repo_root / "src"),
            str(repo_root),
            env.get("PYTHONPATH", ""),
        ]
    ).strip(os.pathsep)

    command = [
        "arch",
        "-x86_64",
        str(python_exe),
        str(driver),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0 and "Remote MVMs are disabled for this session" in completed.stderr:
        pytest.skip(completed.stderr.strip())

    assert completed.returncode == 0, completed.stderr or completed.stdout
    payload = json.loads(result_json.read_text(encoding="utf-8"))
    assert payload["sqrt_result"] == pytest.approx(4.0)
    assert payload["script_output_exists"] is True
    assert payload["script_payload"]["sqrt_via_script"] == pytest.approx(7.0)
    assert payload["script_payload"]["marker"] == "matlab-script-ok"
