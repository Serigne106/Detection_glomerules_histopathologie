# Pipeline complet. Les données du challenge vont dans ./Set02_Glom_train_clean/{train03,train_AKI}
SEEDS ?= 42 43 44
PY ?= python

.PHONY: install test smoke train-ref train-all evaluate compare export benchmark api demo docker

install:
	pip install -e ".[api,onnx,demo,viz,dev]"

test:
	pytest -q

smoke:                     ## vérification rapide sur les vraies données (quelques minutes)
	$(PY) scripts/train.py --config configs/smoke.yaml --out runs/smoke
	$(PY) scripts/evaluate.py --checkpoint runs/smoke/model.pth --out reports_smoke

train-ref:                 ## référence, 3 seeds (≈ 3 h par seed sur CPU)
	for s in $(SEEDS); do $(PY) scripts/train.py --config configs/unet_scratch.yaml --seed $$s; done

train-all: train-ref       ## + modèle pré-entraîné et ablations (1 seed chacun)
	$(PY) scripts/train.py --config configs/unet_pretrained.yaml --seed 42
	for c in ablation_no_color ablation_no_scale ablation_no_quality; do \
		$(PY) scripts/train.py --config configs/$$c.yaml --seed 42; done

evaluate:
	$(PY) scripts/evaluate.py --checkpoint runs/*/model.pth --out reports

compare:
	$(PY) scripts/compare.py --reports reports --out reports/comparison.md
	$(PY) scripts/make_figures.py --runs runs --reports reports --out docs/figures

export:
	$(PY) scripts/export_onnx.py --checkpoint runs/unet_scratch_seed42/model_deploy.pth

benchmark:
	$(PY) scripts/benchmark.py --checkpoint runs/unet_scratch_seed42/model_deploy.pth \
		--onnx runs/unet_scratch_seed42/model_deploy.onnx --out reports/benchmark.json

api:
	MODEL_PATH=runs/unet_scratch_seed42/model_deploy.pth uvicorn app.api:app --port 8000

demo:
	MODEL_PATH=runs/unet_scratch_seed42/model_deploy.pth $(PY) app/gradio_app.py

docker:
	mkdir -p models && cp runs/unet_scratch_seed42/model_deploy.pth models/
	docker build -t glomseg . && docker run -p 8000:8000 -v $$PWD/models:/app/models glomseg
