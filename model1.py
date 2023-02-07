import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random
import requests
from pathlib import Path
from tqdm.auto import tqdm
from helper_functions import accuracy_fn
from helper_functions import print_train_time
from timeit import default_timer as timer


train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

device = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32

# Dataset Used to Train the Model
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

# Dataset Used to Test the Model
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

class Model(nn.Module):
    def __init__(self,
                input_shape: int,
                hidden_units:int,
                output_shape: int):

        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape, out_features=hidden_units),
            nn.Linear(in_features=hidden_units,out_features=output_shape)
        )
    def forward(self,x):
        return self.layer_stack(x)

model = Model(input_shape=784,hidden_units=2,output_shape=len(train_data.classes)).to(device)

if Path("helper_functions.py").is_file():
    print("already downloaded")
else:
    print("Downloading helper_functions.py script")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f:
        f.write(request.content)


LEARNING_RATE = 0.05

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model.parameters(),lr=LEARNING_RATE)

train_time_start_on_cpu = timer()
EPOCHS = 8

for epoch in tqdm(range(EPOCHS)):
    print(f"Epoch: {epoch}")
    train_loss = 0

    for batch, (X,y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        model.train()
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss+=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss/=len(train_dataloader)

    test_loss, test_acc = 0,0
    model.eval()

    with torch.inference_mode():
        for batch, (X_test,y_test) in enumerate(test_dataloader):
            X_test, y_test = X_test.to(device), y_test.to(device)
            test_pred = model(X_test)
            test_loss+=loss_fn(test_pred, y_test)
            test_acc += accuracy_fn(y_true=y_test,y_pred=test_pred.argmax(dim=1))
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"Train loss: {train_loss:.4f} | Test loss: {test_loss:.4f} | Accuracy: {test_acc:.4f}\n")

train_time_end_on_cpu = timer()

total_train_time_model = print_train_time(start=train_time_start_on_cpu,end=train_time_end_on_cpu,device=str(next(model.parameters()).device))

def eval_model(model: torch.nn.Module,data_loader: torch.utils.data.DataLoader,loss_fn:torch.nn.Module,accuracy_fn):
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            # Make predictions with the model
            y_pred = model(X)
            
            # Accumulate the loss and accuracy values per batch
            loss += loss_fn(y_pred, y)
            acc += accuracy_fn(y_true=y, 
                                y_pred=y_pred.argmax(dim=1)) # For accuracy, need the prediction labels (logits -> pred_prob -> pred_labels)
        
        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(data_loader)
        acc /= len(data_loader)
        
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}

# Calculate model 0 results on test dataset
model_results = eval_model(model=model, data_loader=test_dataloader,
    loss_fn=loss_fn, accuracy_fn=accuracy_fn
)
print(model_results)

def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
  pred_probs = []
  model.eval()
  with torch.inference_mode():
      for sample in data:
          # Prepare sample
          sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

          # Forward pass (model outputs raw logit)
          pred_logit = model(sample)

          # Get prediction probability (logit -> prediction probability)
          pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)

          # Get pred_prob off GPU for further calculations
          pred_probs.append(pred_prob.cpu())
          
  # Stack the pred_probs to turn list into a tensor
  return torch.stack(pred_probs)

def plot_predictions(test_samples, test_labels):
    # Plot predictions
    plt.figure(figsize=(9, 9))
    nrows = 3
    ncols = 3
    for i, sample in enumerate(test_samples):
    # Create a subplot
        plt.subplot(nrows, ncols, i+1)

        # Plot the target image
        plt.imshow(sample.squeeze(), cmap="gray")

        # Find the prediction label (in text form, e.g. "Sandal")
        pred_label = train_data.classes[pred_classes[i]]

        # Get the truth label (in text form, e.g. "T-shirt")
        truth_label = train_data.classes[test_labels[i]] 

        # Create the title text of the plot
        title_text = f"Pred: {pred_label} | Truth: {truth_label}"
        
        # Check for equality and change title colour accordingly
        if pred_label == truth_label:
            plt.title(title_text, fontsize=10, c="g") # green text if correct
        else:
            plt.title(title_text, fontsize=10, c="r") # red text if wrong
        plt.axis(False);
    plt.show()


test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs= make_predictions(model=model, 
                             data=test_samples)
# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)

# Make Predictions
plot_predictions(test_samples, test_labels)