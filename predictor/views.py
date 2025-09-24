from django.shortcuts import render
from django.core.files.storage import default_storage
from django.conf import settings
from PIL import Image
import torch
from torchvision import transforms, models
import torch.nn as nn
import os

# Load model once when server starts
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 1)
model.load_state_dict(torch.load(os.path.join('model','ic_real_fake_resnet18.pth'), map_location=device))
model.to(device)
model.eval()

# Transform for input images
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img)
        prob = torch.sigmoid(output).item()
    if prob >= 0.5:
        return f"Real ({prob*100:.2f}%)"
    else:
        return f"Fake ({(1-prob)*100:.2f}%)"

def home(request):
    result = None

    # List of sample images directly in static/
    sample_images = ['test1.jpg','test2.jpg','test3.jpg','test4.jpg','test5.jpg','test6.jpg']

    if request.method == 'POST':
        # Uploaded file
        if request.FILES.get('file'):
            img_file = request.FILES['file']
            path = default_storage.save('tmp/'+img_file.name, img_file)
            img_path = os.path.join('tmp', img_file.name)
            result = predict_image(img_path)
        # Sample image clicked
        elif request.POST.get('sample_file'):
            sample_file = request.POST.get('sample_file')
            img_path = os.path.join(settings.BASE_DIR, 'predictor', 'static', sample_file)
            result = predict_image(img_path)

    return render(request, 'home.html', {'result': result, 'sample_images': sample_images})
