import subprocess
import sys

PACKAGES = [
    "numpy",
    "torch",
    "torchvision",
    "matplotlib",
    "transformers",
]

def main():
    print("Installation des dépendances manquantes...")
    for pkg in PACKAGES:
        print(f"  pip install {pkg}")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  [ERREUR] {pkg} :\n{result.stderr.strip()}")
        else:
            print(f"  [OK] {pkg}")
    print("\nInstallation terminée. Relancez alvis.py.")

if __name__ == "__main__":
    main()
