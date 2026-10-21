from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent

SCRIPTS = [
    "01_ml_baselines_and_splits.py",
    "02_course_heterogeneity.py",
    "03_multitask_erm_and_domain_generalization.py",
    "04_domain_invariance_diagnostic.py",
]

for script in SCRIPTS:
    path = HERE / script
    print(f"\n=== Running {script} ===")
    subprocess.run([sys.executable, str(path)], check=True)

print("\nCore supervised-learning and robustness experiment scripts completed.")
