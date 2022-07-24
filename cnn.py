import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
from torch.utils.data import DataLoader
from dataset import PedestrianStreetDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
in_channel = 3
num_classes = 3
num_epochs = 6
batch_size = 32
learning_rate = 0.0005

# load data
dataset = PedestrianStreetDataset(csv_file = 'data.csv', root_dir = 'data', transform = transforms.ToTensor())


train_set, test_set = torch.utils.data.random_split(dataset, [1008, 100])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

#model
model = torchvision.models.googlenet(pretrained=True)
model.to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

#train network
for epoch in range(num_epochs):
    losses = []
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        #get data to cuda
        data = data.to(device=device)
        targets = targets.to(device=device)
        
        #forward
        scores = model(data)
        loss = criterion(scores, targets)
        
        losses.append(loss.item())
        
        #backward
        optimizer.zero_grad()
        loss.backward()
        
        optimizer.step()
                
print('Finished Training')

# Accuracy
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100}')

    model.train()
    
print("Checking accuracy")

check_accuracy(test_loader, model)
