from torchvision import datasets, transforms
from model import res_net_model
from torch.utils.data import DataLoader,Subset
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=res_net_model().to(device)


data_transforms=transforms.Compose([
    transforms.Resize((64,64)),#eсли железо норм то 224
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),  
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    
    transforms.Normalize((0.5,), (0.5,))
])
val_transforms=transforms.Compose([
   transforms.Resize((64,64)), 
   transforms.ToTensor(),
   
   transforms.Normalize((0.5,), (0.5,))
])
data_dirr='./data'
full_train_dataset=datasets.ImageFolder(root=data_dirr,transform=data_transforms)
full_val_dataset=datasets.ImageFolder(root=data_dirr,transform=val_transforms)

dataset_size=len(full_train_dataset)
train_size= int(0.8*dataset_size)
val_size=dataset_size-train_size

indices=torch.randperm(dataset_size).tolist()

train_dataset=Subset(full_train_dataset,indices[:train_size])
val_dataset=Subset(full_val_dataset,indices[train_size:])




train_loader=DataLoader(train_dataset,batch_size=16,shuffle=True)
val_loader=DataLoader(val_dataset,batch_size=16,shuffle=False)

criterion=nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

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
    scheduler.step(val_acc)
torch.save(model.state_dict(), "face_model.pth")
print("\nОбучение завершено! Файл 'face_model.pth' сохранен.")
 