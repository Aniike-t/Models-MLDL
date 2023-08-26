import torch
from torch import nn
import matplotlib.pyplot as plt


torch.__version__

weight=0.7
bias=0.3

start = 0
end =1
step =0.02
X= torch.arange(start,end,step).unsqueeze(dim=1)
Y = weight*X+bias

X[:10],Y[:10]
len(X),len(Y)

train_split= int(0.8*len(X))
X_train,Y_train = X[:train_split],Y[:train_split]
X_test,Y_test = X[train_split:],Y[train_split:]

len(X_train),len(X_test)


def plot_predictions (train_data=X_train,
                      train_labels=Y_train,
                      test_data=X_test,
                      test_labels=Y_test,
                      predictions=None):
  plt.figure(figsize=(10,7))
  plt.scatter(train_data,train_labels,c="b",s=4,label="Training data")
  plt.scatter(test_data,test_labels,c="g",s=4,label="Testing data")
  if predictions is not None:
    plt.scatter(test_data, predictions.detach().numpy(), c="r", label="predictions")
  plt.legend(prop={"size":14});

plot_predictions();


#Model does is : Takes random values of of weight and bias in the start
#looks at training data and adjusts the randn values and gets closer to te ideal values/dataset values
#Algos : Gradient descent and back propagation

class LinearRegresssionModel(nn.Module):
  #nn.model -->pytorch models
  def __init__(self):
    super().__init__()

    #model parameters (variables used by model)
    self.weight = nn.Parameter(torch.randn(1,
                                            requires_grad=True,   #grad=true bcoz of gardient descent
                                            dtype=torch.float))
    self.bias = nn.Parameter(torch.randn(1,
                                        requires_grad=True,
                                        dtype=torch.float))



  def forward(self, x: torch.Tensor) -> torch.Tensor: #<- x is the input data
    return self.weight*x + self.bias    #formula


torch.manual_seed(42) #for flavoured seed
model_0 =LinearRegresssionModel()
print(list(model_0.parameters()))


y_preds = model_0(X_test)
print(y_preds)

plot_predictions(predictions=y_preds)



#setup loss func
loss_fn=nn.L1Loss()

#setup optm -->popular -->SGD stochastic gardient descent
optimizer = torch.optim.SGD(params=model_0.parameters(),  #parameters in our model
                            lr=0.01)                      #lr -->rate defines the change made at a single step




#an epoch is one loop through the data
epochs=200

epoch_count=[]
loss_values=[]
test_loss_values=[]

#0.loop the data
for epoch in range(epochs):

  #put model ino training mode
  model_0.train()
  
  #forward pass
  y_pred=model_0(X_train)

  #calculate loss
  loss=loss_fn(y_pred,Y_train)

  #optimizer zero gard
  optimizer.zero_grad()

  #perform back propagation to check prev values
  loss.backward()

  #step the optimizer perform gradient descent
  optimizer.step()

  #testing
  model_0.eval() #turns off gradient tracking
  with torch.inference_mode(): #same and better to .no_gard
    test_pred=model_0(X_test)
    test_loss=loss_fn(test_pred,Y_test)
    
    epoch_count.append(epoch)
    loss_values.append(loss)
    test_loss_values.append(test_loss)

  if epoch%10==0:
    print(f"Epoch : {epoch} | loss : {loss} | test_loss : {test_loss}")
    print(model_0.state_dict())


with torch.inference_mode():
  y_preds_new=model_0(X_test)
  print(y_preds_new)


plot_predictions(predictions=y_preds_new);


#plot the loss curves
import numpy as np

plt.plot(epoch_count,np.array(torch.tensor(loss_values).cpu().numpy()),label="train loss")
plt.plot(epoch_count, test_loss_values,label="Test Loss")
plt.title("Training and test loss curves")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend();