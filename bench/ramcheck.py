"""How much memory does it take to *serve* an out-of-core index?

Builds an index of N documents, then reopens it in a fresh process and reports
current RSS at each stage, so growth can be attributed rather than guessed.
"""

import os
import resource
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, "/mnt/dados3/sandbox/arara-rag")

CACHE = Path("/mnt/dados3/sandbox/arara-rag/.cache/ramcheck")
SENT = [
    "A alíquota do imposto de renda é progressiva e varia conforme a faixa.",
    "O licenciamento ambiental é um instrumento preventivo da política nacional.",
    "A CONTRATADA obriga-se a manter sigilo sobre as informações a que tiver acesso.",
    "O prazo para interposição de recurso é de quinze dias úteis contados da intimação.",
    "Compete ao órgão ambiental estadual o licenciamento de impacto regional.",
]


def corpus(n):
    import numpy as np

    rng = np.random.default_rng(0)
    return {
        f"d{i:06d}": " ".join(str(SENT[j]) for j in rng.integers(0, len(SENT), size=4))
        for i in range(n)
    }


def rss_mb():
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return 0.0


def peak_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build(n):
    from arara_rag import Arara

    path = CACHE / f"idx{n}"
    import shutil

    shutil.rmtree(path, ignore_errors=True)
    a = Arara(path=path)
    a.add_documents(corpus(n))
    a.finalize()
    stats = a.stats()
    a.close()
    return stats


def serve(n):
    from arara_rag import Arara

    path = CACHE / f"idx{n}"
    cap = os.environ.get("ARARA_MAX_RAM_MB")
    cap = float(cap) if cap else None
    base = rss_mb()
    a = Arara(path=path, max_ram_mb=cap)
    opened = rss_mb()
    a.search("imposto de renda", top_k=10)
    one = rss_mb()
    for _ in range(20):
        a.search("licenciamento ambiental", top_k=10)
    many = rss_mb()
    st = a.stats()
    a.close()
    return {
        "documents": n,
        "chunks": st["chunks"],
        "rss_before_mb": round(base, 1),
        "rss_after_open_mb": round(opened, 1),
        "rss_after_1_query_mb": round(one, 1),
        "rss_after_20_queries_mb": round(many, 1),
        "rss_to_serve_mb": round(many - base, 1),
        "peak_rss_to_serve_mb": round(peak_mb() - base, 1),
        "vector_bytes": st["vector_bytes"],
        "lexical_bytes": st["lexical_bytes"],
        "slots_bytes": st["slot_bookkeeping_bytes"],
        "max_ram_mb": cap,
        "scan_block": st.get("scan_block"),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: ramcheck.py {build|serve|run} <sizes...>")
        raise SystemExit(2)
    if sys.argv[1] == "build":
        print(build(int(sys.argv[2])))
    elif sys.argv[1] == "serve":
        import json

        print(json.dumps(serve(int(sys.argv[2]))))
    else:
        import json

        CACHE.mkdir(parents=True, exist_ok=True)
        rows = []
        for n in (int(x) for x in sys.argv[2:]):
            subprocess.run([sys.executable, __file__, "build", str(n)], check=True)
            out = subprocess.run(
                [sys.executable, __file__, "serve", str(n)],
                check=True, capture_output=True, text=True,
            )
            rec = json.loads(out.stdout.strip().splitlines()[-1])
            rows.append(rec)
            print(json.dumps(rec), flush=True)
        (CACHE / "results.json").write_text(json.dumps(rows, indent=2))
