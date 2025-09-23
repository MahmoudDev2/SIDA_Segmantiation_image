import torch
import torchvision.transforms as transforms
from PIL import Image
from .resnet import resnet50  # Use relative import

def load_model(model_path, use_cpu=True):
    """Loads the pre-trained CNNDetection model."""
    model = resnet50(num_classes=1)

    device = 'cuda' if not use_cpu and torch.cuda.is_available() else 'cpu'
    state_dict = torch.load(model_path, map_location='cpu')

    # The model state dict is nested under the 'model' key
    model.load_state_dict(state_dict['model'])
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model, device

def predict_image(model, device, image_path, crop_size=None):
    """
    Predicts if an image is real or fake using the loaded model.
    Returns the probability of the image being synthetic.
    """
    # Define image transformations
    trans_init = []
    if crop_size:
        trans_init.append(transforms.CenterCrop(crop_size))

    trans = transforms.Compose(trans_init + [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Open and transform the image
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = trans(img).unsqueeze(0)
    except Exception as e:
        print(f"Error opening or transforming image: {e}")
        return None

    # Perform prediction
    with torch.no_grad():
        img_tensor = img_tensor.to(device)
        prob = model(img_tensor).sigmoid().item()

    return prob
