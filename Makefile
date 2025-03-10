0-create-venv:
    python3 -m venv .venv
#make sure you source the environment before running the next command
# this does not work
1-activate-venv:
    source .venv/bin/activate
2-install-required-packages:
    pip install -r requirements.txt
