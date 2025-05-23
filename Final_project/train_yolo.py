from ultralytics import YOLO
import argparse
import os


def train_yolo(args):   # Create a new YOLO model from scratch
    if args.weights:
        # Load custom pretrained weights
        model = YOLO(args.weights)
    else:
        # Create a new YOLO model from scratch
        model = YOLO("yolo11m.yaml")
        # Load default pretrained YOLO model
        model = YOLO("yolo11m.pt")

    # Train the model
    results = model.train(
        data=args.data,  # path to dataset.yaml
        epochs=args.epochs,
        imgsz=args.img_size,
        batch=args.batch_size,
        name=args.name,
        device=args.device,
        workers=args.workers,
        pretrained=args.pretrained,
        project=args.save_dir,
        exist_ok=True
    )

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Export the model to ONNX format
    success = model.export(format="onnx")


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO model on LiveCell dataset')
    parser.add_argument('--data', type=str, default='data/livecell_yolo/dataset.yaml',
                        help='path to dataset.yaml file')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--img_size', type=int, default=640,
                        help='input image size')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--name', type=str, default='yolo_livecell',
                        help='experiment name')
    parser.add_argument('--device', type=str, default="0",
                        help='device to use (e.g., "0" for GPU 0, "" for CPU)')
    parser.add_argument('--workers', type=int, default=8,
                        help='number of worker threads')
    parser.add_argument('--pretrained', default=True, action='store_true',
                        help='use pretrained weights')
    parser.add_argument('--save_dir', type=str, default='/mnt/sda1/cv/yolo/livecell-2',
                        help='directory to save the trained model')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to custom pretrained weights file (.pt)')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    train_yolo(args)


if __name__ == '__main__':
    main()
