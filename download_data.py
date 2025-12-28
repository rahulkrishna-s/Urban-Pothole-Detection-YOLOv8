import os
from dotenv import load_dotenv
from roboflow import Roboflow

def download_dataset():
    # load env variable
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY")

    if not api_key:
        print("Error. API key not found.")
        return

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("myworkspace-9h67e").project("pothole_segmentation_yolov8-i0izs")
    version = project.version(1)

    print("Downloading dataset..")
    dataset = version.download("yolov8", location="datasets/pothole_dataset")
    if dataset:
        print("Success")

if __name__ == "__main__":
    download_dataset()
