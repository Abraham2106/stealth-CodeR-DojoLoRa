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
    import os
    import platform
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    return Path, json, mo, os, platform, shutil, subprocess, sys


@app.cell
def _(mo):
    mo.md("""
    # CodeR DojoLoRa · preparación

    Revisor local con un backbone Qwen, LoRA A para reviews y LoRA B para riesgo/calidad.
    **Entrenamiento bloqueado. No se actualizarán pesos.**

    Este notebook utiliza datos públicos de DojoCodingLabs, incluidos reviews de bots
    identificados como supervisión débil. La calidad real y el límite local de 4 GB
    todavía no están certificados. Nunca pegues tokens en el código ni en las salidas.
    """)
    return


@app.cell
def _(json, mo):
    import urllib.request as _summary_request
    try:
        with _summary_request.urlopen(
            "https://raw.githubusercontent.com/Abraham2106/stealth-CodeR-DojoLoRa/main/reports/preparation-summary.json", timeout=20
        ) as _summary_response:
            public_summary = json.load(_summary_response)
        _summary_display = mo.vstack([mo.md("## Evidencia de preparación · corpus público"),
                   mo.ui.table([public_summary["collection"]]),
                   mo.md("Todos los reviews de bots se mantienen como supervisión débil; ninguna cifra implica calidad del modelo."),
                   mo.ui.table([public_summary.get("preparation") or {"status": "pending"}])])
    except Exception as _summary_error:
        public_summary = None
        _summary_display = mo.md(f"Resumen remoto todavía no disponible: {type(_summary_error).__name__}.")
    _summary_display
    return (public_summary,)


@app.cell
def _(mo):
    mo.md("""
    ## Estado de ejecución

    Habilitar RTX Pro 6000 en los recursos del notebook permite comprobar CUDA.
    Cambiar tablas o figuras no ejecuta entrenamiento. La sesión y sus archivos
    generados pueden expirar: los artefactos deben exportarse explícitamente.
    """)
    return


@app.cell
def _(json, platform, shutil, subprocess):
    import importlib.metadata as metadata
    _packages = {}
    for _package in ("marimo", "torch", "transformers", "peft", "bitsandbytes"):
        try:
            _packages[_package] = metadata.version(_package)
        except metadata.PackageNotFoundError:
            _packages[_package] = None
    _gpu = None
    if shutil.which("nvidia-smi"):
        _gpu_result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20,
        )
        _gpu = _gpu_result.stdout.strip() if _gpu_result.returncode == 0 else "nvidia-smi failed"
    runtime_report = {
        "python": platform.python_version(), "gpu_observed": _gpu, "packages": _packages,
        "training_executed": False, "deployment_4gb_verified": False,
    }
    print(json.dumps(runtime_report, indent=2))
    return (runtime_report,)


@app.cell
def _(mo):
    bootstrap = mo.ui.run_button(label="Preparar código y dependencias (sin entrenamiento)")
    mo.vstack([mo.md("Se descargará una revisión inmutable del repo personal; esa revisión quedará registrada."), bootstrap])
    return (bootstrap,)


@app.cell
def _(bootstrap, mo, subprocess, sys):
    mo.stop(not bootstrap.value, mo.md("La instalación está pendiente de su botón explícito."))
    import urllib.request
    import json as bootstrap_json
    with urllib.request.urlopen("https://api.github.com/repos/Abraham2106/stealth-CodeR-DojoLoRa/commits/main", timeout=30) as _response:
        executed_code_revision = bootstrap_json.load(_response)["sha"]
    _requirement = f"coder-dojolora @ git+https://github.com/Abraham2106/stealth-CodeR-DojoLoRa.git@{executed_code_revision}"
    subprocess.run([sys.executable, "-m", "pip", "install", _requirement], check=True, capture_output=True, text=True)
    mo.md(f"Código preparado: `{executed_code_revision}`. No se ha entrenado ningún modelo.")
    return (executed_code_revision,)


@app.cell
def _(executed_code_revision, mo):
    from coder.model import PROFILES
    from coder.evaluation import preflight
    prepared_preflight = preflight()
    prepared_preflight["code_revision"] = executed_code_revision
    mo.vstack([mo.md("## Perfiles candidatos (sin medir)"), mo.ui.table(list(PROFILES.values())),
               mo.ui.table([prepared_preflight])])
    return (prepared_preflight,)


@app.cell
def _(mo):
    collect_button = mo.ui.run_button(label="Recopilar historial público de Dojo (sin entrenar)")
    collect_button
    return (collect_button,)


@app.cell
def _(collect_button, executed_code_revision, mo):
    mo.stop(not collect_button.value, mo.md("El corpus no se descarga automáticamente al abrir el notebook."))
    from coder.data import collect
    collected_report = collect("data", workers=4)
    mo.ui.table([collected_report])
    return (collected_report,)


@app.cell
def _(mo):
    context_button = mo.ui.run_button(label="Preparar snapshots, grafo y splits auditados")
    context_button
    return (context_button,)


@app.cell
def _(collected_report, context_button, mo):
    mo.stop(not context_button.value, mo.md("La construcción histórica espera su botón explícito."))
    from coder.data import prepare
    prepared_report = prepare("data", with_context=True)
    mo.ui.table([prepared_report])
    return (prepared_report,)


@app.cell
def _(Path, json, mo, prepared_report):
    _examples = json.loads(Path("data/prepared/train.json").read_text(encoding="utf-8"))
    _sample = [{"task": e["task"], "repo": e["repo"], "pr": e["pr"],
                "author_type": e["provenance"]["author_type"], "target": e["target"]} for e in _examples[:6]]
    mo.vstack([mo.md("## Ejemplos · supervisión sin verificar"), mo.ui.table(_sample)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Puerta de entrenamiento: cerrada

    No existe un botón activo que entrene en esta etapa. Antes de habilitarlo hacen falta:
    presupuesto explícito de horas/pasos/gasto, revisión del corpus y sus licencias,
    etiquetas suficientes por tarea, revisión inmutable del modelo y configuración elegida.

    Baselines, ablaciones, calibración, evaluación humana, VRAM y latencia quedan
    preparados en el paquete; **no hay métricas de modelos entrenados ni adaptadores entrenados**.
    """)
    return


if __name__ == "__main__":
    app.run()
