import mlflow.pyfunc
from rag_pipeline import build_qa_chain

class RAGModel(mlflow.pyfunc.PythonModel):

    def load_context(self, context):
        self.qa_chain = build_qa_chain(
            persist_dir=context.artifacts["chroma_db"],
            prompt_path=context.artifacts["prompt"]
        )

    def predict(self, model_input):
        question = model_input.iloc[0]["question"]
        result = self.qa_chain({"query": question})
        return result["result"]
