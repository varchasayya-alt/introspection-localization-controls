"""Download the 20 layer-2 concept vectors used in Studies 3-5 from the authors' repository
(github.com/elyhahami18/llama-introspection-new @ 7d9280d, MIT licence) into
$INTROSPECTION_DATA/study3/their_vectors and verify each file's SHA-256 against
stimuli/study4_stimuli.json.

Usage: python scripts/download_their_vectors.py
"""
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import introspect_lib as L  # noqa: E402

COMMIT = "7d9280d79e4aa37599dc0fd89974be4bc7a54827"
URL = "https://raw.githubusercontent.com/elyhahami18/llama-introspection-new/{c}/data/saved_vectors/llama/{f}"


def main():
    stim = json.loads((L.REPO_ROOT / "stimuli" / "study4_stimuli.json").read_text(encoding="utf-8"))
    out = L.DATA_ROOT / "study3" / "their_vectors"
    out.mkdir(parents=True, exist_ok=True)
    for f, h in stim["vector_sha256"].items():
        dest = out / f
        if not dest.exists():
            with urllib.request.urlopen(URL.format(c=COMMIT, f=f)) as r:
                dest.write_bytes(r.read())
        got = hashlib.sha256(dest.read_bytes()).hexdigest()
        if got != h:
            raise SystemExit(f"SHA-256 mismatch for {f}: {got} != {h}")
        print(f"ok  {f}")
    print(f"all {len(stim['vector_sha256'])} files verified in {out}")


if __name__ == "__main__":
    main()
