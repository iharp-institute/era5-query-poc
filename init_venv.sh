rm -rf venv

python3.11 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip setuptools

python -m pip install wheel

python -m pip install -r requirements.txt