import mlflow
import mlflow.pyfunc
from rag.rag_model import RAGModel
import os

mlflow.set_experiment("RAG_Experiment")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path="rag_model",
        python_model=RAGModel(),
        artifacts={
            "chroma_db": os.path.join(BASE_DIR, "../../data/chroma_db"),
            "prompt": os.path.join(BASE_DIR, "./prompts/rag_prompt.txt")
        },
        registered_model_name="RAG_Pipeline"
    )
