#!/usr/bin/env bash
# Prépare le dossier du Space : bash hf_space/deploy.sh runs/unet_scratch_seed42/model_deploy.pth
set -euo pipefail
MODEL=${1:?chemin du model_deploy.pth}
OUT=build_space
rm -rf "$OUT" && mkdir -p "$OUT/models" "$OUT/examples"
cp hf_space/{app.py,requirements.txt,README.md} "$OUT/"
cp -r src/glomseg "$OUT/"
cp app/gradio_app.py "$OUT/"
cp "$MODEL" "$OUT/models/model_deploy.pth"
cp examples/*.png "$OUT/examples/" 2>/dev/null || echo "(pas d'exemples : ajoutez des images libres de droits dans examples/)"
echo "Space prêt dans $OUT/ -> git push vers https://huggingface.co/spaces/<compte>/glomeruli-finder"
