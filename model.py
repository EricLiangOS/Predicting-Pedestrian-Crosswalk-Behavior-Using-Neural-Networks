import torch
import torch.nn as nn
import torch.optim as optim
import torchvision


class Model():
    def __init__(self, in_channels, num_classes, num_epochs, batch_size, learning_rate):
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
        self.model = torchvision.models.googlenet(pretrained=True)
        self.model.to(self.device)
    
    def train(self, dataset):
        
        train_set = dataset
        train_loader = torch.utils.data.DataLoader(train_set, batch_size= self.batch_size, shuffle=True)

        # loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        #train network
        for epoch in range(self.num_epochs):
            losses = []
            
            for batch_idx, (data, targets) in enumerate(train_loader):
                #get data to cuda
                data = data.to(device=self.device)
                targets = targets.to(device=self.device)
                
                #forward
                scores = self.model(data)
                loss = criterion(scores, targets)
                
                losses.append(loss.item())
                
                #backward
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
    
    def get_model(self):
        return self.model
    
    def evaluate(self, img):
        
        self.model.eval()
        
        with torch.no_grad():
            scores = self.model(img)
            
            _, prediction = scores.max(1)
            return prediction.item()
        

