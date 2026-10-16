# Point d'entrée Hugging Face Spaces. Contenu du Space :
#   app.py, requirements.txt, README.md, glomseg/ (copie de src/glomseg),
#   gradio_app.py (copie de app/gradio_app.py), models/model_deploy.pth, examples/*.png
import runpy

runpy.run_path("gradio_app.py", run_name="__main__")
