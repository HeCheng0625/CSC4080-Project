from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

path_img = 'train_images/ff631653374e.png'
img = Image.open(path_img).convert('RGB')
img = transform(img).view(1, 3, 224, 224)
img = img.to(device)
# print(img)

predict_model = models.resnet18()
predict_model.fc = nn.Linear(512, 5)

predict_model.to(device)
predict_model.load_state_dict(torch.load('resnet18_pretrained'))
# print(predict_model)

predict_model.eval()
with torch.no_grad():
    output = predict_model(img)
    probabilities = F.softmax(output, dim=1).view(-1).cpu().numpy().tolist()
    print(probabilities)
    _, predicted = torch.max(output.data, 1)
    predicted = predicted.view(-1).cpu().numpy().tolist()
    print(predicted)