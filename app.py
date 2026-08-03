import os
import torch
import torch.nn as nn
from PIL import Image
from flask import Flask, render_template, request
from torchvision import transforms

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

classes = [
    "Deer",
    "Elephant",
    "Horse",
    "Monkey",
    "Tiger",
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


class AnimalCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


model = AnimalCNN()

checkpoint = torch.load(
    "models/checkpoint.pth",
    map_location=torch.device("cpu")
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        uploads_FOLDER,
        file.filename
    )

    file.save(filepath)

    image = Image.open(filepath).convert("RGB")
    image = transform(image)
    image = image.unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, 1).item()

    return render_template(
        "index.html",
        prediction=classes[prediction],
        image_path=filepath
    )


if __name__ == "__main__":
    app.run(debug=True)