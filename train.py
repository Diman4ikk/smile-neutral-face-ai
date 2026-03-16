from torchvision import datasets, transforms
from model import SimpleCNN
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=SimpleCNN().to(device)


data_transforms=transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Normalize((0.5,), (0.5,))
])

data_dirr='./data'
dataset=datasets.ImageFolder(root=data_dirr,transform=data_transforms)

train_size=int(0.8*len(dataset))
val_size=len(dataset)-train_size
train_dataset,val_dataset=torch.utils.data.random_split(dataset,[train_size,val_size])


train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=16,shuffle=False)

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
    model.eval()
    val_correct=0
    with torch.no_grad():
          for images,labes in val_loader:
            images,labes=images.to(device),labes.to(device)
            outputs=model(images)
            prediction=torch.argmax(outputs,dim=1)
            val_correct+=(prediction==labes).sum().item()
    average_loss=running_loss/len(train_loader)
    accuracy=corretc/total
    val_acc = val_correct / len(val_dataset)
    print(f"Epoch {epoch+1}, Loss: {average_loss:.4f}, Accutacy:{accuracy:.4f}| Val Acc: {val_acc:.4f}")

torch.save(model.state_dict(), "face_model.pth")
print("\nОбучение завершено! Файл 'face_model.pth' сохранен.")