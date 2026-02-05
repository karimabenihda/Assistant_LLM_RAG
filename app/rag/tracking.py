import mlflow
import mlflow.pyfunc
from rag_model import RAGModel
import os

# Configuration de l'URI de tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("RAG_Experiment")

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../"))
DB_PATH = os.path.join(ROOT_DIR, "mlflow.db")

mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")

with mlflow.start_run() as run:
    # Chemins absolus pour éviter les surprises
    artifacts = {
        "chroma_db": os.path.abspath(os.path.join(BASE_DIR, "../../data/chroma_db")),
        "prompt": os.path.abspath(os.path.join(BASE_DIR, "../prompts/rag_prompt.txt"))
    }
    
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        code_paths=[
            os.path.join(BASE_DIR, "rag_model.py"),
            os.path.join(BASE_DIR, "rag_pipeline.py")
        ],
        artifacts=artifacts,
        registered_model_name="RAG_Pipeline"
    )
    print(f"Modèle enregistré avec succès ! Run ID: {run.info.run_id}")
    
    
    