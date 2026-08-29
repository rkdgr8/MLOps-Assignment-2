import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class PneumoniaModel:
    def __init__(self, model_path="models/model.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])
        
        self.model = models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        
        try:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            logger.info("Model weights loaded successfully.")
        except Exception as e:
            logger.warning(f"Failed to load specific weights (using untrained): {e}")
            
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_bytes):
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            input_tensor = self.preprocess(image)
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_batch)
                prob = torch.nn.functional.softmax(output[0], dim=0)
                conf, predicted = torch.max(prob, 0)
                
            result = self.classes[predicted.item()]
            return {
                "prediction": result,
                "confidence": float(conf.item()),
                "probabilities": {
                    self.classes[0]: float(prob[0].item()),
                    self.classes[1]: float(prob[1].item())
                }
            }
        except Exception as e:
            logger.error(f"Prediction logic failed: {e}")
            raise e
