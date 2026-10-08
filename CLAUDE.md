# Primera Competencia DMEyF 2026 — mapa del repo

Repo propio de la competencia (antes vivía en `dmeyf2026/monday/competencia/`, el fork de la cátedra).
Remoto `origin` = `git@github.com:agustinarazanov/dmeyf.git`; **se commitea y pushea directo a `main`**
(git plano; GitButler quedó instalado pero el workspace se desarmó con `but teardown` el 8-oct).
Cierra el **sábado 10 de octubre 23:59 ART según el reloj del bot** (`2026-10-11T03:01:01` UTC);
Zulip dice domingo 11. Se entrega el sábado.

El diario completo de decisiones, retractaciones y mediciones está en `docs/diario_competencia.md`
(1.600 líneas; es el `CLAUDE.md` histórico). Este archivo es solo el mapa y las reglas.

## Dónde está cada cosa

```
competencia-1/
├── README.md                 el reporte: modelo final, cómo reproducir, qué se probó y qué dio
├── CLAUDE.md                 este mapa
├── reproducir.py             del crudo al CSV entregado, bit a bit, con las semillas
├── competencia.py            datos, ganancia, corte, caché de modelos, escritura de envíos
├── registro.py               el ledger en Postgres (base `competencia`, contenedor zulip-postgres)
├── enviar.py                 ledger -> bot -> respuesta -> ledger, un submit por comando
├── fe_panel.py               panel base + generadores de SQL (lags, deltas, ranks, ventanas)
├── scripts/cNNN_*.py         un experimento por script, numerados; tmp_recuperados/ los que
│                             estaban solo en /tmp (c114–c190)
├── experimentos/<cNNN>/      modelos (.txt), scores (.parquet), envios/*.csv   <- solo los CSV y
│                             json/md se versionan; modelos y scores pesan 4 GB y se regeneran
├── ledger/                   export CSV de las tablas del ledger (experimento, submit, archivo,
│                             resultado), para que la trazabilidad viaje con el repo
├── docs/                     diario, auditorías, diagnóstico de FE, propuestas de features
└── data -> ../dmeyf2026/monday/data   symlink; los parquet no se versionan
```

Las clases y los notebooks de la cátedra siguen en `../dmeyf2026/monday/` (ver su `CLAUDE.md`).
El archivo de Zulip está en `../zulip/` (Postgres `zulip`, mismo contenedor). Las notas de clase y
los apuntes por notebook, en el vault de Obsidian `DMEyF/`.

## Reglas de la casa (no negociables)

1. **Un submit mide una hipótesis con fundamento**, escrita antes en el ledger con su delta contra una
   referencia. Nunca al azar, nunca para "dejar vigente" el mejor puntaje: eso se hace solo cuando la
   usuaria lo indique, cerca del cierre. El cupo diario se gasta entero midiendo: **17** desde el 8-oct (el bot pasó de x/13 a x/17 sin anuncio); si rechaza el 14º, volver a 13. Renueva 21:00 ART.
2. **Nada se envía sin pasar por `enviar.py`**, que registra primero (`estado='preparado'`) y envía
   después. Sin `--enviar` solo registra e imprime el resumen para aprobar.
3. **Todo lo que cuesta cómputo se persiste**: modelos por semilla en `experimentos/<exp>/modelo_*.txt`,
   scores de 202108 por semilla en `scores_202108_s<semilla>.parquet`. Re-cortar es gratis.
4. **Cada submit es reproducible**: dataset, params, meses, target, semillas y corte quedan en el
   ledger; el CSV se escribe ordenado por id para que un `diff` verifique la reproducción.
5. **`es`/`y` salen SIEMPRE del mismo DataFrame que las predicciones.** DuckDB no garantiza el orden
   de las filas; ya costó un submit con ganancia −49,86 y dos mediciones imposibles.
6. **No sacar variables para predecir 202108.** Préstamos incluidos. Lo dijo el profesor con sus
   experimentos (Zulip 2026-10-05) y lo confirmaron tres mediciones públicas.
7. **Comparaciones públicas con 20 archivos de una semilla**, mismas semillas y mismo corte que la
   referencia. Ruido de partición ±3 M entre modelos, ±4 M entre cortes anidados: reportar la
   diferencia en clientes (÷1,1 M), el sd por sorteo y el z.

## Entorno

pyenv `facultad` (Python 3.14) para todo lo de modelos; pyenv `zulip` para `enviar.py` (tiene el
módulo `zulip` además de `psycopg`). Después de un reboot: `open -a Docker && docker start zulip-postgres`.

Las pistas del profesor, con ids de Zulip, están en `docs/pistas_profesor.md`.

## Estado

Ver `README.md` para el modelo vigente y la tabla pública. Última novedad: `c201_receta_lags`
(lags/deltas 1 y 2 de todo + receta de Denicolay) cruzó los 100 públicos; su óptimo de corte está en
la banda 13.000–14.500, no en 9.000–10.000 como la base.
