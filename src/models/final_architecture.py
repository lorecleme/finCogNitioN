class CNN(nn.Module):
    
    
    def __init__(self, num_classes=30):
        super().__init__()
        
        self.ConvLayer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU())
        
        self.ConvLayer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d((2,2)),
            nn.ReLU())
        
        self.ConvLayer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.ConvLayer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2,2)),
            nn.ReLU())
        
        self.ConvLayer5 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),   
            nn.ReLU())
        
        self.ConvLayer6 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2,2)),
            nn.ReLU())
        
       
        
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(12544, 2048),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(2048, 256),
            nn.ReLU())
        self.fc3 = nn.Sequential(
            nn.Linear(256, num_classes))
        
    
    
    def forward(self, inputs):
        
        x = self.ConvLayer1(inputs)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.ConvLayer4(x)
        x = self.ConvLayer5(x)
        x = self.ConvLayer6(x)

        x = x.reshape(x.size(0), -1)
        
        
        x = self.fc1(x)
        x = self.fc2(x)
        
        return self.fc3(x)