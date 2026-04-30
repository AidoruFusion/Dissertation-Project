
import subprocess
import sys
import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = [
    "ethereum/adversarial/ethereum_svm_adv_model.py",
    "ethereum/adversarial/ethereum_lr_adv_model.py",
    "ethereum/adversarial/ethereum_knn_adv_model.py",
    "ethereum/adversarial/ethereum_rf_adv_model.py",
    "ethereum/adversarial/ethereum_mlp_adv_model.py",
]


def run_script(script_name):
    script_path = os.path.join(BASE_DIR, script_name)

    if not os.path.exists(script_path):
        print(f"\n[SKIPPED] {script_name} was not found.")
        return

    print("\n" + "=" * 80)
    print(f"Running {script_name}")
    print("=" * 80)

    result = subprocess.run(
        [sys.executable, script_path],
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        print(f"\n[ERROR] {script_name} failed.")
        sys.exit(result.returncode)

    print(f"\n[DONE] {script_name}")


def main():
    for script in SCRIPTS:
        run_script(script)

    print("\n" + "=" * 80)
    print("All Ethereum adversarial scripts finished successfully.")
    print("Check the figures folder for the confusion matrices.")
    print("=" * 80)


if __name__ == "__main__":
    main()