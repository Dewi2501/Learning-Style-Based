from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_DATA = ROOT / "DATA" / "public_release" / "moodle_learner_activity.csv.gz"
DERIVED = ROOT / "DATA" / "derived"
OUT = ROOT / "OUTPUT" / "recomputed_public"
