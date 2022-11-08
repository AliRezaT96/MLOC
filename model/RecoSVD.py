from surprise import SVD, Dataset
from surprise.accuracy import rmse
from surprise.dump import dump

# Load the movielens-100k dataset (download it if needed).
data = Dataset.load_builtin('ml-100k')
trainset = data.build_full_trainset()
trainset.head()

# Use an example algorithm: SVD.
algo = SVD()
algo.fit(trainset)

# predict ratings for all pairs (u, i) that are in the training set.
testset = trainset.build_testset()
predictions = algo.test(testset)
rmse(predictions)                                                                              

#actual predictions as thse items have not been seen by the users. there is no ground truth. 
# We predict ratings for all pairs (u, i) that are NOT in the training set.
testset = trainset.build_anti_testset()
predictions = algo.test(testset)

dump('./surprise_model', predictions, algo)

