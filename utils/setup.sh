pip install -r requirements.txt

cd d4rl
pip install -e .
cd ../rl-toolkit
pip install -e .
cd ..

mkdir -p data/trained_models
pip install wandb -U