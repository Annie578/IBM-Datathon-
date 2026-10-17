#!/usr/bin/env bash
# One-time setup on the LinuxONE VM (Ubuntu 24.04, s390x).
# Run from the repo root:  bash scripts/setup_vm.sh
# Installs packages one at a time so a single failure does not hide the rest.
set -u

sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-dev build-essential git

# --system-site-packages lets us fall back to apt-provided numpy/pandas/etc. if pip cannot build them
python3 -m venv .venv --system-site-packages
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip

: > setup_failures.txt
for pkg in numpy pandas scipy scikit-learn pyarrow matplotlib joblib xgboost shap; do
  echo "=== installing $pkg"
  if ! pip install "$pkg"; then
    echo "FAILED: $pkg" | tee -a setup_failures.txt
  fi
done

echo
if [ -s setup_failures.txt ]; then
  echo "Some packages failed:"; cat setup_failures.txt
  echo "Fallback for numpy/pandas/scipy/sklearn:  sudo apt-get install -y python3-numpy python3-pandas python3-scipy python3-sklearn"
else
  echo "All packages installed."
fi

echo
echo "Next: source .venv/bin/activate && python scripts/env_check.py"
