"""Un submit, de punta a punta y con trazabilidad: ledger -> bot -> respuesta -> ledger.

    python enviar.py --experimento c201_receta_lags --submit c201_lags_10000 \
        --archivos 'experimentos/c201_receta_lags/envios_10000/*_s*.csv' \
        --hipotesis "..." --delta-contra c191_base_10000 --delta-desc "..." \
        [--enviar]

Sin --enviar solo registra el experimento y el submit en estado 'preparado' e
imprime el resumen: nombre, archivos, cantidad de envios, referencia. La regla de
la casa es que nada se manda sin mostrar eso antes. Con --enviar sube los
archivos con la cuenta de la alumna, espera la respuesta del bot, la guarda en
zulip/respuestas/<message_id>.json y la registra en `resultado`.
"""
import argparse
import glob
import importlib.util
import subprocess
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent     # la raiz del repo
sys.path.insert(0, str(RAIZ / "src"))
import registro as r  # noqa: E402

ZULIP = Path.home() / "Documents/segundo-cuatrimestre/zulip/competencia.py"


def modulo_zulip():
    sys.path.insert(0, str(ZULIP.parent))      # competencia.py del repo zulip importa main.py
    spec = importlib.util.spec_from_file_location("zulip_competencia", ZULIP)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=RAIZ, text=True).strip()
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experimento", required=True)
    ap.add_argument("--submit", required=True)
    ap.add_argument("--archivos", required=True, help="glob de los CSV")
    ap.add_argument("--hipotesis", required=True)
    ap.add_argument("--delta-contra", default=None)
    ap.add_argument("--delta-desc", default=None)
    ap.add_argument("--descripcion", default=None)
    ap.add_argument("--corte", type=int, default=None)
    ap.add_argument("--notas", default=None)
    ap.add_argument("--final", action="store_true")
    ap.add_argument("--enviar", action="store_true")
    ap.add_argument("--reenviar", action="store_true",
                    help="permite reenviar un submit que ya figura como enviado/respondido")
    ap.add_argument("--esperar", type=int, default=240)
    args = ap.parse_args()

    rutas = sorted(Path(p) for p in glob.glob(args.archivos))
    if not rutas:
        sys.exit(f"ningun archivo coincide con {args.archivos!r}")
    archivos = []
    for ruta in rutas:
        n = sum(1 for l in ruta.read_text().splitlines() if l.strip())
        sem = ruta.stem.rsplit("_s", 1)[-1] if "_s" in ruta.stem else None
        archivos.append({"nombre": ruta.name, "ruta": ruta.resolve(),
                         "semilla": int(sem) if sem and sem.isdigit() else None,
                         "n_envios": n})

    with r.conectar() as con:
        r.crear_esquema(con)
        exp_id = r.alta_experimento(
            con, args.experimento, notebook=f"{args.experimento}.py", git_sha=git_sha(),
            descripcion=args.descripcion, corte_envios=args.corte,
            n_semillas=sum(1 for a in archivos if a["semilla"] is not None) or None,
            ruta=str((RAIZ / "experimentos" / args.experimento).resolve()))
        existe = con.execute("select id, estado from submit where nombre=%s",
                             (args.submit,)).fetchone()
        if existe is None:
            r.alta_submit(con, args.submit, exp_id, archivos, hipotesis=args.hipotesis,
                          delta_contra=args.delta_contra, delta_descripcion=args.delta_desc,
                          es_final=args.final, notas=args.notas)
            print(f"registrado submit {args.submit} (experimento {args.experimento}, id {exp_id})")
        else:
            print(f"el submit {args.submit} ya existe en estado {existe['estado']!r}")
            if args.enviar and existe["estado"] in ("enviado", "respondido") and not args.reenviar:
                sys.exit("  ya fue enviado: reenviarlo gasta cupo y choca con resultado.unique; "
                         "usar --reenviar si es a proposito")
        con.commit()

    print(f"  {len(archivos)} archivos, envios por archivo: "
          f"{sorted({a['n_envios'] for a in archivos})}")
    print(f"  hipotesis: {args.hipotesis}")
    print(f"  delta contra: {args.delta_contra} | {args.delta_desc}")
    if not args.enviar:
        print("  (no enviado: falta --enviar)")
        return

    z = modulo_zulip()
    client = z.cliente()
    res = z.enviar_submit(client, args.submit, rutas)
    with r.conectar() as con:
        r.marcar_enviado(con, args.submit, res["message_id"])
        con.commit()
    print(f"  enviado, message_id={res['message_id']}; esperando al bot hasta {args.esperar}s")
    mensajes = z.escuchar(client, desde_id=res["message_id"], espera=args.esperar, cola=res["cola"])
    if not mensajes:
        print("  el bot no respondio a tiempo; correr `competencia.py ultimas` y registrar a mano")
        return
    z.guardar(mensajes)
    # resultado tiene unique(submit_id): se registra UNA respuesta, la primera con numeros
    # (o la primera a secas si ninguna los trae, que es el caso de rechazo).
    con_numeros = [m for m in mensajes if z.parsear(m)["public_gain_mean"] is not None]
    for m in (con_numeros or mensajes)[:1]:
        p = z.parsear(m)
        with r.conectar() as con:
            r.alta_resultado(con, args.submit, p["respuesta_cruda"],
                             public_gain_mean=p["public_gain_mean"],
                             public_gain_std=p["public_gain_std"],
                             n_archivos=p["n_archivos"], submits_usados=p["submits_usados"])
            if not p["aceptado"]:
                r.marcar_estado(con, args.submit, "fallido")
            con.commit()
        print(f"  bot: mean {p['public_gain_mean']}  std {p['public_gain_std']}  "
              f"archivos {p['n_archivos']}  cupo {p['submits_usados']}/{p['submits_tope']}  "
              f"aceptado={p['aceptado']}")


if __name__ == "__main__":
    main()
