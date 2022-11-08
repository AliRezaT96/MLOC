from surprise import Dataset
from surprise.dump import load
from collections import defaultdict
import pandas as pd
import flask

def get_top_n(predictions, n=10):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n


df = pd.read_csv('./movies.dat',sep="::",header=None,engine='python')
df.columns = ['iid','name','genre']
df.set_index('iid',inplace=True)
predictions, algo = load('./surprise_model')
top_n = get_top_n(predictions, n=5)
app = flask.Flask(__name__)

# define a predict function as an endpoint
@app.route("/", methods=["GET"])
def predict():
    data = {"success": False}


    # check for passed in parameters   
    params = flask.request.json
    if params is None:
        params = flask.request.args
    
    if "uid" in params.keys(): 
        data["response"] = str([df.loc[int(iid),'name'] for (iid, _) in top_n[params.get("uid")]])
        data["success"] = True
        
    # return a response in json format 
    return flask.jsonify(data)


# start the flask app, allow remote connections
app.run(host='0.0.0.0')
