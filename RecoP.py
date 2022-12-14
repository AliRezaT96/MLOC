# https://github.com/NicolasHug/Surprise
from surprise import SVD, Dataset
from surprise.accuracy import rmse
from surprise.dump import dump
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Loss, MeanSquaredError
from datetime import datetime
from sklearn.utils import shuffle


class Loader():
    current = 0

    def __init__(self, x, y, batchsize=1024, do_shuffle=True):
        self.shuffle = shuffle
        self.x = x
        self.y = y
        self.batchsize = batchsize
        self.batches = range(0, len(self.y), batchsize)
        if do_shuffle:
            # Every epoch re-shuffle the dataset
            self.x, self.y = shuffle(self.x, self.y)

    def __iter__(self):
        # Reset & return a new iterator
        self.x, self.y = shuffle(self.x, self.y, random_state=0)
        self.current = 0
        return self

    def __len__(self):
        # Return the number of batches
        return int(len(self.x) / self.batchsize)

    def __next__(self):
        n = self.batchsize
        if self.current + n >= len(self.y):
            raise StopIteration
        i = self.current
        xs = torch.from_numpy(self.x[i:i + n])
        ys = torch.from_numpy(self.y[i:i + n])
        self.current += n
        return (xs, ys)



def l2_regularize(array):
    loss = torch.sum(array ** 2.0)
    return loss


class MF(nn.Module):
    itr = 0
    
    def __init__(self, n_user, n_item, k=18, c_vector=1.0, c_bias=1.0):
        super(MF, self).__init__()
        self.k = k
        self.n_user = n_user
        self.n_item = n_item
        self.c_bias = c_bias
        self.c_vector = c_vector
        
        self.user = nn.Embedding(n_user, k)
        self.item = nn.Embedding(n_item, k)
        
        # We've added new terms here:
        self.bias_user = nn.Embedding(n_user, 1)
        self.bias_item = nn.Embedding(n_item, 1)
        self.bias = nn.Parameter(torch.ones(1))
    
    def __call__(self, train_x):
        user_id = train_x[:, 0]
        item_id = train_x[:, 1]
        vector_user = self.user(user_id)
        vector_item = self.item(item_id)
        
        # Pull out biases
        bias_user = self.bias_user(user_id).squeeze()
        bias_item = self.bias_item(item_id).squeeze()
        biases = (self.bias + bias_user + bias_item)
        
        ui_interaction = torch.sum(vector_user * vector_item, dim=1)
        
        # Add bias prediction to the interaction prediction
        prediction = ui_interaction + biases
        return prediction
    
    def loss(self, prediction, target):
        loss_mse = F.mse_loss(prediction, target.squeeze())
        
        # Add new regularization to the biases
        prior_bias_user =  l2_regularize(self.bias_user.weight) * self.c_bias
        prior_bias_item = l2_regularize(self.bias_item.weight) * self.c_bias
        
        prior_user =  l2_regularize(self.user.weight) * self.c_vector
        prior_item = l2_regularize(self.item.weight) * self.c_vector
        total = loss_mse + prior_user + prior_item + prior_bias_user + prior_bias_item
        return total

def log_training_loss(engine, log_interval=400):
    epoch = engine.state.epoch
    itr = engine.state.iteration
    fmt = "Epoch[{}] Iteration[{}/{}] Loss: {:.2f}"
    msg = fmt.format(epoch, itr, len(train_loader), engine.state.output)
    model.itr = itr
    if itr % log_interval == 0:
        print(msg)

def log_validation_results(engine):
    evaluat.run(test_loader)
    metrics = evaluat.state.metrics
    avg_accuracy = metrics['accuracy']
    print("Epoch[{}] Validation MSE: {:.2f} "
          .format(engine.state.epoch, avg_accuracy))
    
    
#Data
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
uir = np.array([x for x in trainset.all_ratings()])
train_x = test_x = uir[:,:2].astype(np.int64)
train_y = test_y = uir[:,2].astype(np.float32)


#Parameters
lr = 1e-2
k = 10 #latent dimension
c_bias = 1e-6
c_vector = 1e-6
batchsize = 1024

model = MF(trainset.n_users, trainset.n_items, k=k, c_bias=c_bias, c_vector=c_vector)
optimizer = torch.optim.Adam(model.parameters())
trainer = create_supervised_trainer(model, optimizer, model.loss)
metrics = {'accuracy': MeanSquaredError()}

evaluat = create_supervised_evaluator(model, metrics=metrics)
train_loader = Loader(train_x, train_y, batchsize=batchsize)
test_loader = Loader(test_x, test_y, batchsize=batchsize)
trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED, handler=log_training_loss)
trainer.add_event_handler(event_name=Events.EPOCH_COMPLETED, handler=log_validation_results)

trainer.run(train_loader, max_epochs=50)
torch.save(model.state_dict(), "./pytorch_model/model.pt")
