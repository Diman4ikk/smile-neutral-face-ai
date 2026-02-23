from torchvision import datasets, transforms
from model import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=SimpleCNN().to(device)


data_transforms=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Grayscale()
])

data_dirr='./data'
dataset=datasets.ImageFolder(root=data_dirr,transform=data_transforms)
train_loader=DataLoader(dataset,batch_size=16,shuffle=True)

criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for epoch in range(15):
    model.train()
    running_loss=0.0
    corretc=0
    total=0

    for images ,labes in train_loader:
            images,labes=images.to(device),labes.to(device)

            optimizer.zero_grad()
            outputs=model(images)
            loss=criterion(outputs,labes)
            loss.backward()
            optimizer.step()

            running_loss+=loss.item()

            prediction=torch.argmax(outputs,dim=1)
            corretc+=(prediction==labes).sum().item()
            total+=labes.size(0)

    average_loss=running_loss/len(train_loader)
    accuracy=corretc/total
    print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, Accutacy:{accuracy:.4f}")

torch.save(model.state_dict(), "face_model.pth")
print("\nОбучение завершено! Файл 'face_model.pth' сохранен.")