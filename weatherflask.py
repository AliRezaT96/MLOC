
# load Flask 
import flask
from flask import jsonify
from geopy.geocoders import Nominatim
import requests

app = flask.Flask(__name__)

# define a predict function as an endpoint 
@app.route("/", methods=["GET","POST"])
def weather():

    data = {"success": False}
    #https://pypi.org/project/geopy/
    geolocator = Nominatim(user_agent="cloud_function_weather_app")


    params = flask.request.json
    if params is None:
        params = flask.request.args


    # params = request.get_json()
    if "msg" in params:
        location = geolocator.geocode(str(params['msg']))
        # https://www.weather.gov/documentation/services-web-api
        result1 = requests.get(f"https://api.weather.gov/points/{location.latitude},{location.longitude}")
        result2 = requests.get(f"{result1.json()['properties']['forecast']}")
        data["response"] = result2.json()
        data["success"] = True
    return jsonify(data)
    
# start the flask app, allow remote connections
if __name__ == '__main__':
    app.run(host='0.0.0.0')


