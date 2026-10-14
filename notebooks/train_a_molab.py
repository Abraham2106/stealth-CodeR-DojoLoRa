# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo==0.25.1"]
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import json
    import subprocess
    import sys
    import time
    import shutil
    import io
    import zipfile
    from pathlib import Path
    return Path, io, json, mo, shutil, subprocess, sys, time, zipfile


@app.cell
def _(mo):
    mo.md("""
    # CodeR · entrenar LoRA A

    **Alcance autorizado:** solo A, reviews de bots como supervisión débil;
    B permanece sin entrenar. Límite total: **6 horas GPU, 2.000 updates incluyendo
    el smoke, o 3 épocas; gasto externo $0**. No se afirma calidad técnica ni
    compatibilidad medida con 4 GB.

    El entrenamiento tiene un botón explícito y no se reinicia al actualizar gráficos.
    """)
    return


@app.cell
def _(Path, json, shutil, subprocess, time):
    _gpu = None
    if shutil.which("nvidia-smi"):
        _query = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=20)
        _gpu = _query.stdout.strip() if _query.returncode == 0 else None
    _state = Path(".coder")
    _state.mkdir(exist_ok=True)
    _budget_file = _state / "gpu-budget.json"
    if _gpu and not _budget_file.exists():
        _budget_file.write_text(json.dumps({"started_at": time.time(), "limit_seconds": 21600, "gpu": _gpu}))
    gpu_started_at = json.loads(_budget_file.read_text())["started_at"] if _budget_file.exists() else None
    gpu_observed = _gpu
    print(json.dumps({"gpu_observed": gpu_observed, "gpu_started_at": gpu_started_at,
                      "training_started": False, "budget_seconds": 21600}, indent=2))
    return gpu_observed, gpu_started_at


@app.cell
def _(mo):
    bootstrap_button = mo.ui.run_button(label="Preparar entorno CUDA y código")
    corpus_upload = mo.ui.file(filetypes=[".zip"], multiple=False, label="Subir coder-corpus.zip auditado")
    mo.vstack([bootstrap_button, corpus_upload])
    return bootstrap_button, corpus_upload


@app.cell
def _(Path, bootstrap_button, gpu_observed, mo, subprocess, sys):
    mo.stop(not bootstrap_button.value, mo.md("El entorno espera su botón explícito."))
    mo.stop(not gpu_observed, mo.md("Habilita la RTX 6000 antes de preparar el entorno."))
    import urllib.request as _bootstrap_request
    import json as _bootstrap_json
    with _bootstrap_request.urlopen("https://api.github.com/repos/Abraham2106/stealth-CodeR-DojoLoRa/commits/main", timeout=30) as _response:
        executed_revision = _bootstrap_json.load(_response)["sha"]
    _venv = Path(".coder/venv").resolve()
    if not _venv.exists():
        subprocess.run([sys.executable, "-m", "venv", str(_venv)], check=True)
    trainer_python = str(_venv / "bin/python")
    _log_path = Path(".coder/setup.log")
    with _log_path.open("w") as _log:
        subprocess.run([trainer_python, "-m", "pip", "install", "torch==2.11.0", "--index-url", "https://download.pytorch.org/whl/cu128"],
                       stdout=_log, stderr=subprocess.STDOUT, check=True, timeout=1800)
        _package = f"coder-dojolora[model,evaluation] @ git+https://github.com/Abraham2106/stealth-CodeR-DojoLoRa.git@{executed_revision}"
        subprocess.run([trainer_python, "-m", "pip", "install", _package],
                       stdout=_log, stderr=subprocess.STDOUT, check=True, timeout=1800)
    _preflight_code = "import json; from coder.evaluation import preflight; print(json.dumps(preflight(), indent=2))"
    runtime_preflight = subprocess.run([trainer_python, "-c", _preflight_code], check=True, capture_output=True, text=True).stdout
    print(runtime_preflight)
    mo.md(f"Entorno listo. Revisión ejecutada: `{executed_revision}`. Aún no se entrena.")
    return executed_revision, runtime_preflight, trainer_python


@app.cell
def _(Path, corpus_upload, io, json, mo, zipfile):
    mo.stop(not corpus_upload.value, mo.md("Sube el corpus auditado para habilitar el entrenamiento."))
    dataset_directory = Path(".coder/data").resolve()
    dataset_directory.mkdir(parents=True, exist_ok=True)
    _required = {"train.json", "validation.json", "test.json", "preparation-report.json"}
    with zipfile.ZipFile(io.BytesIO(corpus_upload.contents())) as _archive:
        if not _required.issubset(_archive.namelist()):
            raise ValueError("Corpus incompleto")
        if sum(_archive.getinfo(_name).file_size for _name in _required) > 512 * 1024 * 1024:
            raise ValueError("Corpus excede el tamaño autorizado para este importador")
        for _name in _required:
            (dataset_directory / _name).write_bytes(_archive.read(_name))
    uploaded_audit = json.loads((dataset_directory / "preparation-report.json").read_text())
    if not uploaded_audit.get("passed"):
        raise ValueError("La auditoría del corpus no pasó")
    mo.ui.table([uploaded_audit])
    return dataset_directory, uploaded_audit


@app.cell
def _(mo):
    train_button = mo.ui.run_button(label="Entrenar LoRA A · máximo 6 h / 2000 updates / 3 épocas")
    refresh = mo.ui.refresh(options=[10, 30], default_interval=10)
    mo.vstack([train_button, refresh])
    return refresh, train_button


@app.cell
def _(Path, dataset_directory, executed_revision, gpu_started_at, json, mo, subprocess, time, train_button, trainer_python, uploaded_audit):
    mo.stop(not train_button.value, mo.md("Entrenamiento no iniciado."))
    mo.stop(gpu_started_at is None, mo.md("No se observó GPU."))
    if time.time() - gpu_started_at >= 21600 - 300:
        raise TimeoutError("Presupuesto GPU agotado")
    _marker = Path(".coder/training-started.json")
    if _marker.exists():
        raise RuntimeError("Ya se inició una corrida en esta sesión; no se reinicia automáticamente")
    run_directory = Path(".coder") / ("run-A-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()))
    _marker.write_text(json.dumps({"output": str(run_directory), "code_revision": executed_revision,
                                   "gpu_started_at": gpu_started_at, "trained_adapters": ["A"]}))
    _log = Path(".coder/training.log").open("w")
    training_process = subprocess.Popen([trainer_python, "-u", "-m", "coder.train_a", "--data", str(dataset_directory),
                                         "--output", str(run_directory), "--gpu-started-at", str(gpu_started_at), "--profile", "qwen35"],
                                        stdout=_log, stderr=subprocess.STDOUT)
    mo.md(f"Corrida iniciada. PID `{training_process.pid}`. Se guarda en `{run_directory}`.")
    return run_directory, training_process


@app.cell
def _(Path, json, mo, refresh, run_directory, training_process):
    refresh
    mo.stop("run_directory" not in globals(), mo.md("El monitor espera una corrida explícita."))
    _log_path = Path(".coder/training.log")
    _lines = _log_path.read_text(errors="replace").splitlines()[-18:] if _log_path.exists() else []
    _report_path = run_directory / "report.json"
    if _report_path.exists():
        training_report = json.loads(_report_path.read_text())
        _display = mo.vstack([mo.md("## Resultado observado"), mo.ui.table([training_report]),
                              mo.md("B sigue sin entrenar. NLL no certifica corrección técnica."),
                              mo.md("```text\n" + "\n".join(_lines) + "\n```")])
    else:
        training_report = None
        _display = mo.vstack([mo.md(f"Estado del proceso: `{training_process.poll()}` (`None` = activo)."),
                              mo.md("```text\n" + "\n".join(_lines) + "\n```")])
    _display
    return (training_report,)


@app.cell
def _(Path, mo, refresh, run_directory, training_report):
    refresh
    mo.stop(training_report is None, mo.md("La descarga aparecerá al finalizar el guardado y la evaluación."))
    _archive_path = Path(str(run_directory.resolve()) + ".zip")
    mo.stop(not _archive_path.exists(), mo.md("Creando archivo de exportación…"))
    mo.download(data=_archive_path.read_bytes(), filename="coder-lora-A-trained.zip", mimetype="application/zip",
                label="Descargar LoRA A, tokenizer, métricas y manifiestos")
    return


if __name__ == "__main__":
    app.run()
