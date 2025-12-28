from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO('yolov8n.pt')

    # Train model
    results = model.train(
        data='datasets/pothole_dataset/data.yaml', 
        epochs=25, 
        imgsz=640,
        name='pothole_v1' 
    )

if __name__ == '__main__':
    main()