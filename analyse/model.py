from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import ImageDraw


class ImageProcessor:
    def __init__(self, model_url="facebook/detr-resnet-50", revision="no_timm"):
        self.processor = DetrImageProcessor.from_pretrained(model_url, revision=revision)
        self.model = DetrForObjectDetection.from_pretrained(model_url, revision=revision)

    def process(self, image, threshold=0.9):
        if image.format == "PNG":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([image.size[::-1]])
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        results = list(zip(results["scores"], results["labels"], results["boxes"]))

        # Create a draw object
        draw = ImageDraw.Draw(image)

        for i, (score, label, box) in enumerate(results):
            box = [round(coord, 2) for coord in box.tolist()]
            label_text = self.model.config.id2label[label.item()]

            # Draw a rectangle around the detected object
            draw.rectangle(box, outline="red", width=3)

            # Replace the label in results with the corresponding text
            results[i] = (score, label_text, box)

        return image, results


image_processor = ImageProcessor("facebook/detr-resnet-50")
