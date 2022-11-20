from flask import Flask, request # import flask class, request module (to accept user inputs)
import pandas as pd # to work with dataframe
import os # to get port number that we'll hard-code to 8080  for now - important for deployment on Google Cloud (port no. will not be hard-coded there and sends a variable called port!)
import mlflow.pyfunc # to load downloaded model for making predictions (same as mlops-1 'Making Predictions using the local downloaded model')
import joblib
from io import StringIO


# Step 2: Instantiate the Flask API with name 'ModelEndpoint' ('api' is an object of the Flask() class)
api = Flask('ModelEndpoint') # 'ModelEndpoint' can be called anything else as well

# Step 3: Load the model from best_estimator folder for subsequently making predictions
# model = mlflow.pyfunc.load_model(model_uri="./best_estimator") # same code as mlops-1 'Making Predictions using the local downloaded model')
# doing this as the above line does not work
model_classify = joblib.load("./best_estimator/model.pkl")

# Step 4: Create the routes (we can create multiple! similar to multiple functions in a Python script)
# Note: we'll need to name each route differently, similar to naming individual functions differently in a Python script

## route 1: Health check. Just return success if the API is running
@api.route('/') # this is a decorator (@api - using the Flask class' object instantiated above and creating a 'route' on this 'api' flask object. then we pass a name for this route as '/' just means home page)
def home(): # just a normal Python function called 'home' with a decorator addition above
    # return a simple string as JSON (JSON is just Python dictionary)
    return {"message": "Hi there!", "success": True}, 200
# returns dictionary with a message, code for success 'True' and html code 200 (https://en.wikipedia.org/wiki/List_of_HTTP_status_codes)
# a user going to the home page of this API, will see this message defined in the dictionary (we'll see this in action soon!)

## route 2: accept input data, convert from JSON to dataframe, run predictions on the df, convert predictions to a list & return as dictionary. flask will take care of conversion to JSON object
# 'POST' method is used when we want to receive some data from the user and POST it to the API. when we want to access the route '/predict' route, we'll always need to post some data to it, else it'll error
@api.route('/predict', methods = ['POST']) # naming this 2nd route as /predict. so, in https://en.wikipedia.org/wiki, this is equivalent of the '/wiki', while the home page is what comes before '/wiki'
def make_predictions(): # create a normal Python function for predictions
    # step 1: Get the JSON object input data sent over the API
    user_input = request.get_json(force=True) # use 'request' module (imported from Flask earlier)'s .get_json method
    # by setting force=True in request.get_json, Flask will auto route input data sent to '/predict' route onto variable 'user_input'
    import sys
    # print("***********************")
    # print(type(user_input), user_input, file = sys.stderr)
    # step 2: Convert user inputs (JSON object from step#1) to pandas dataframe
    df_schema = {"article":str} # To ensure the feature columns for modeling get the correct datatype of float, because when Pandas converts from JSON to df, it infers dtype of every col
    user_input_df = pd.read_json(StringIO(user_input), lines=True, dtype=df_schema) # Convert JSONL to dataframe with additional argument of dtype of what we're expecting the API to handle so model predictions work fine
    print("***********************")
    print(type(user_input_df), user_input_df, file = sys.stderr)
    # step 3: Run predictions using the loaded 'model' on user_input_df and convert predictions output from numpy array to list
    predictions = model_classify.predict(pd.Series(user_input_df["article"][0])).tolist()
    
    if predictions[0]==1:
        return{'prediction': f'This is a FAKE news'}
    else:
        return{'prediction': f'This is a REAL news'}
    
    #return {'predictions': predictions} # return output of 'predict' route as a dictionary for Flask to convert to JSON object & send back to user at the '/predict' route. dictionary's key (can be any name) as 'predictions', values as list of model predictions
    

# Step 5: Main function that actually runs the API! - simply (blindly) copy+paste for all API runs
if __name__ == '__main__': # good practise to have this main block whenever creating a .py file
    api.run(host='0.0.0.0', # run the 'api' object created above with 2 routes on local host url '0.0.0.0' to just run on this computer
            debug=True, # Debug=True ensures any changes to inference.py (like adding an extra print somewhere in this script) automatically updates the running API
            port=int(os.environ.get("PORT", 8080)) # just use 8080 by default
           ) 
