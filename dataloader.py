import torch 
from torch_geometric.data import Data, Batch
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn 
from torch import optim
from models import Combined , final
from train_eval import train_model , evaluate_model

def create_graph_data(features, labels=None):
    num_nodes = features.size(0)
    edge_index = torch.combinations(torch.arange(num_nodes), r=2).t()
    x = features.clone().detach()
    y = labels.clone().detach() if labels is not None else None
    return Data(x=x, edge_index=edge_index, y=y)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load the data
datadir = 'C:/Users/aspri/OneDrive/Desktop/college resources/ML DL SHITE/datasets/brain_tumor_dataset/brain_tumor_dataset'
dataset = ImageFolder(datadir, transform=transform)

dataset_size = len(dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.15 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_1 = Combined(num_classes=2)
model_2 = final(num_classes=2)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_2.parameters(), lr=0.001)  


model , history = train_model(train_loader=train_loader , val_loader=val_loader , model=model_2 , criterion=criterion , optimizer=optimizer , num_epochs=10 , device=device)




    

