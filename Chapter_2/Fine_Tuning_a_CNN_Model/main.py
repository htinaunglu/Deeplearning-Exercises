from torchvision import models

def create_model():
    model = models.resnet18(pretrained=True)

    for param in model.parameters(): # This Part is Freezing
        param.requires_grad = False  # the model
    # Connect the CNN output to the input of full-layer-neural-network
    num_features=model.fc.in_features 
    model.fc = nn.Sequential(
                   nn.Linear(num_features, 10)) 
    return model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select gpu for training
print(f"Running on Device {device}")

model=create_model()
model=model.to(device)

# Training loop
def train(model, train_loader, cost, optimizer, epoch):
 model.train()
 for e in range(epoch):
     running_loss=0
     correct=0
     for data, target in train_loader:
         data=data.to(device)
         target=target.to(device)
         optimizer.zero_grad()
         pred = model(data)             #No need to reshape data since CNNs take image inputs
         loss = cost(pred, target)
         running_loss+=loss
         loss.backward()
         optimizer.step()
         pred=pred.argmax(dim=1, keepdim=True)
         correct += pred.eq(target.view_as(pred)).sum().item()
     print(f"Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
         Accuracy {100*(correct/len(train_loader.dataset))}%")

