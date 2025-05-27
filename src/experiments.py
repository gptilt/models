import comet_ml
import os

def start():
    return comet_ml.start(
        api_key=os.environ.get("COMET_API_KEY"),
        project_name="win-probability-estimator",
        workspace="gptilt"
    )
