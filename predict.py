from sqlalchemy import false, true
import torchvision.transforms as transforms
import cv2
from PIL import Image
import time
import winsound

from model import Model
from dataset import PedestrianStreetDataset

def low_beep():
    duration = 500
    freq = 440 
    winsound.Beep(freq, duration)

def high_beep():
    duration = 500
    freq = 540 
    winsound.Beep(freq, duration)

def predict(img, model):

    transform1=transforms.Compose([transforms.Resize((100,100)),transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,))])
    #Turns into batch of size 1
    img=transform1(img)  
    img = img.unsqueeze(0)
    
    result = model.evaluate(img)

    if result == 1:
        return "pedestrian"
    if result == 2:
        return "biker"
    return "street"

def take_photo(model): 
    global prev
    cap = cv2.VideoCapture(1)
    _, img = cap.read()

    #Convert from numpy array to processable image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    im_pil = Image.fromarray(img)

    result = predict(im_pil, model)
    print(result)
    cap.release()
    
    #Play sound when detect person
    if result == "pedestrian" or result == "biker":
        if prev:
            high_beep() #Replace with trigger
            prev = True
            return

        prev = True
        
    low_beep()
    prev = False
    
# Parameters
in_channel = 3
num_classes = 3
num_epochs = 4
batch_size = 32
learning_rate = 0.0005

delay = 2.5

# Set up model
train_set = PedestrianStreetDataset(csv_file = 'data.csv', root_dir = 'data', transform = transforms.ToTensor())
model = Model(in_channels=in_channel, num_classes=num_classes, num_epochs=num_epochs, batch_size=batch_size, learning_rate=learning_rate)
model.train(train_set)

starttime = time.time()
prev = False

# Set up camera
while True:
    take_photo(model)
    
    # Pause between times
    time.sleep(delay)