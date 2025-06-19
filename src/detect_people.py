import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image

def detect_people(image_path, threshold=0.8):
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    # COCO class index for 'person' is 1
    PERSON_CLASS = 1

    # Image preprocessing
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    img_tensor = transform(img)

    # Model expects a list of images
    with torch.no_grad():
        predictions = model([img_tensor])[0]

    people_boxes = []
    for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
        if label.item() == PERSON_CLASS and score.item() >= threshold:
            people_boxes.append(box.tolist())

    return people_boxes

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) < 2:
    #     print("Usage: python detect_people.py <image_path>")
    # else:
    #     boxes = detect_people(sys.argv[1])
    #     print("Detected people bounding boxes:", boxes)

    from torchvision.datasets import VOCDetection
    from PIL import Image
    import os

    # Download VOC 2007 dataset (only a few images for test split)
    dataset = VOCDetection(root="data", year="2007", image_set="test", download=True)

    # Get the first image path
    img_path = dataset.images[0]

    # Now you can use your detect_people function
    from detect_people import detect_people

    boxes = detect_people(img_path)
    print("Detected people bounding boxes:", boxes)
