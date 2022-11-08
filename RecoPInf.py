from surprise import Dataset
import numpy as np
import torch
from torch import nn
import pandas as pd

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

def get_top_n(model,testset,trainset,uid_input,n=10):
    
    preds = []
    try:
        uid_input = int(trainset.to_inner_uid(uid_input))
    except KeyError:
        return preds        

    # First map the predictions to each user.
    for uid, iid, _ in testset: #inefficient
        try:
            uid_internal = int(trainset.to_inner_uid(uid))
        except KeyError:
            continue
        if uid_internal==uid_input:
            try:
                iid_internal = int(trainset.to_inner_iid(iid))
                movie_name = df.loc[int(iid),'name']
                preds.append((iid,movie_name,float(model(torch.tensor([[uid_input,iid_internal]])))))
            except KeyError:
                pass
    # Then sort the predictions for each user and retrieve the k highest ones
    if preds is not None:
        preds.sort(key=lambda x: x[1], reverse=True)
        if len(preds) > n:
            preds = preds[:n]
    return preds



#Data
df = pd.read_csv('./movies.dat',sep="::",header=None,engine='python')
df.columns = ['iid','name','genre']
df.set_index('iid',inplace=True)
print(df.head())
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
testset = trainset.build_anti_testset()


#Parameters
lr = 1e-2
k = 10 #latent dimension
c_bias = 1e-6
c_vector = 1e-6

model = MF(trainset.n_users, trainset.n_items, k=k, c_bias=c_bias, c_vector=c_vector)
model.load_state_dict(torch.load('./pytorch_model'))
model.eval()

# Print the recommended items for each user
limit = 0
for uid,_,_ in testset:
    print('\nUser:',uid)
    seen = [df.loc[int(iid),'name'] for (iid, _) in trainset.ur[int(uid)]]
    if len(seen) > 10: seen = seen[:10]
    print('\tSeen:',seen)
    print('\tRecommendations:',get_top_n(model,testset,trainset,uid,n=10))
    limit+=1
    if limit>3:
        break

        