from ultralytics import YOLO

def main():
    # Load the model
    model = YOLO('yolov8n.pt')

    # Train the model
    # results = model.train(data='data.yaml', epochs=50, imgsz=640)

if __name__ == '__main__':
    main()