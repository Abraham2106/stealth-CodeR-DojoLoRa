# CodeR DojoLoRa

Preparación reproducible de un revisor local de PRs: grafo + recuperación local +
un backbone Qwen con dos LoRA y una cabeza multitarea. **No se ha entrenado un modelo.**
El entrenamiento está bloqueado hasta revisar datos, licencias y presupuesto.

## Instalar y preparar

Python 3.12, Git y acceso de lectura a GitHub. Todo el corpus descargado se guarda
localmente y está excluido de Git; no pegar credenciales en notebooks.

```sh
python -m venv .venv
# Windows: .venv/Scripts/activate ; Linux: source .venv/bin/activate
python -m pip install -e .
coder_data collect --workers 4
coder_data prepare --with-context
coder_data audit
marimo edit notebooks/coder_molab.py
```

La adquisición se puede reanudar desde su caché. El token de `gh` se usa en memoria
en el equipo local; en Molab la adquisición funciona con API pública o una variable
secreta administrada por el usuario, nunca una credencial subida desde el equipo.
La API pública sin autenticar tiene cuota reducida; un error de cuota no certifica
un corpus completo.

## CLI

```sh
review_pr --repo /ruta/repo --pr 12 --prepare-only --output contexto.json
review_pr --repo /ruta/repo --base BASE_SHA --head HEAD_SHA --offline --prepare-only --output contexto.json
```

El segundo comando no necesita red. Para review real se requiere `--checkpoint`
con pesos entrenados, tokenizer, cabeza y calibración de validación. No se publican
comentarios ni se aprueban/mezclan PRs. `comments`, `risk` y `human_review_required`
son `null` en preparación; no son predicciones vacías ni probabilidades inventadas.

## Corpus y límites

Solo repos públicos de DojoCodingLabs, incluido el historial propio de forks.
Reviews de bots se marcan como `weak-bot`. Las aprobaciones/cambios solicitados son
proxies de workflow, no riesgo de defectos verificado. Un comentario ignorado no
se etiqueta como malo. Reacciones positivas son evidencia débil; el resto queda
pendiente en `data/prepared/annotation-queue.json`.

La reconstrucción usa el padre del primer commit disponible del PR y verifica
ancestralidad para el commit original del comentario. No certifica el base exacto
al abrir un PR después de rebases; los ejemplos irrecuperables o anclas inválidas
se excluyen. El contexto de código se toma del base, y estándares del mismo commit.
RAG solo incluye reviews anteriores al ejemplo y excluye reviews del mismo PR.
Los splits son cronológicos por grupo de PR; duplicados transversales se embargan.
Auditorías exactas no certifican independencia semántica.

El grafo soporta Python, JS/JSX, TS/TSX; no pretende resolver despacho dinámico.
Imports, referencias, llamadas no resueltas y archivos omitidos quedan registrados.
SQLite FTS5 evita un modelo adicional de embeddings en la GPU.

Se fija tree-sitter 0.25.2: la versión 0.26.0 produjo un fallo nativo reproducido
durante la reconstrucción. El upstream documenta la regresión de referencias en
[issue #472](https://github.com/tree-sitter/py-tree-sitter/issues/472).
No utilizamos el setter `QueryCursor.match_limit` afectado en 0.25.x.

## Modelos y evaluación

Perfiles: `Qwen/Qwen3.5-4B` con loader textual y `Qwen/Qwen3-4B-Instruct-2507`.
Ambos indican Apache 2.0 en sus model cards. Las APIs y revisiones son verificables
en [Qwen3.5](https://huggingface.co/docs/transformers/en/model_doc/qwen3_5),
[PEFT](https://huggingface.co/docs/peft/en/package_reference/peft_model) y
[bitsandbytes](https://huggingface.co/docs/transformers/en/quantization/bitsandbytes).
La LM head nativa permanece congelada y ligada a embeddings. La cabeza B tiene
3 logits de riesgo y 2 de calidad con pérdidas/softmax separados. Se usa una sola
copia del backbone; caches no se comparten al alternar adaptadores.

```sh
python -m pip install -e '.[model,evaluation]'
python -m unittest discover -s tests -v
marimo check notebooks/coder_molab.py
```

Los tests usan modelos diminutos CPU inicializados de cero; no descargan pesos,
no hacen backward ni ejecutan optimizadores. Son verificación de interfaces y
recarga, no entrenamiento ni evidencia de calidad.

`coder.evaluation` prepara métricas por clase, matriz de confusión, ablaciones,
hojas de evaluación humana, exportación de figuras 300 dpi/PDF/SVG y medición CUDA.
No usar coincidencia textual con un bot como prueba de corrección técnica.
El baseline A deshabilita adaptadores manteniendo contexto idéntico; el baseline
de riesgo aplica reglas de tamaño, con un clasificador pequeño pendiente de ajuste
futuro autorizado. Calibración y selección deben usar validación, nunca test.

## Molab

[Abrir notebook en Molab](https://molab.marimo.io/github/Abraham2106/stealth-CodeR-DojoLoRa/blob/main/notebooks/coder_molab.py)

Crear una copia en la cuenta, habilitar RTX Pro 6000 y revisar el preflight observado.
La copia debe registrar la revisión exacta del código al instalarlo. El notebook
no contiene ningún control que entrene. Cada adquisición o construcción costosa
tiene un botón explícito separado de tablas y figuras. Molab es público por defecto
y los archivos generados no necesariamente persisten; exportar antes de terminar.

La elección final y medición en RTX 2050 de 4 GB y la política CPU/GPU quedan
pendientes. Un pico medido en RTX 6000 no demuestra compatibilidad con 4 GB.

## Corrida autorizada de A

[Notebook de entrenamiento A](https://molab.marimo.io/github/Abraham2106/stealth-CodeR-DojoLoRa/blob/main/notebooks/train_a_molab.py).
La corrida solicitada permite 6 horas totales de GPU, 2.000 updates incluyendo
el smoke, o 3 épocas, con gasto externo $0. Solo se entrena A sobre reviews de
bots; B y su cabeza permanecen pendientes de anotación humana. El notebook de
preparación anterior mantiene su bloqueo.

Subir `artifacts/coder-corpus.zip`, con los tres splits congelados y el informe
de auditoría, y preparar el entorno CUDA mediante su botón. El entrenamiento
comienza únicamente con su botón explícito. La unidad de SFT es un hunk cambiado;
las coordenadas del comentario son targets, no campos de entrada. Se enmascara
el prompt, se conserva intacta la respuesta y se registran exclusiones por
longitud. El corpus positivo no enseña cuándo abstenerse de comentar.

```sh
python -m coder.train_a --data data/prepared --output artifacts/run-A \
  --gpu-started-at TIMESTAMP_UNIX_DE_INICIO_DE_GPU --profile qwen35
```

Se compara NLL de tokens de respuesta con el backbone sin adaptadores usando
el mismo test, seleccionando por validación. Una instancia nueva verifica la
recarga y ejecuta generación cuantizada, registrando JSON válido, VRAM y latencia.
Estas métricas no prueban corrección técnica. El checkpoint marca A entrenado y
B sin entrenar; la CLI del revisor completo exige ambos y sigue bloqueada.
Los resultados y pesos solo existen después de una ejecución observada.
