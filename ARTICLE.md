# Machine learning API using FastAPI

The purpose of this project is to build an API that will be used to interact with a Machine Learning model via internet protocol requests. The API is developed using FastAPI.

Many people ask , “What is an API?” . In simple terms, API is the acronym for Application Programming Interface . This is a software mediator that allows two applications to interact with each other. APIs are an accessible way to extract and share data within and across organisations.

We can find API’s all around us. Each time you use an app, say, Uber or Bolt , or send a mobile payment, or change the thermostat temperature from your phone, you’re using an API.

When you use one of the above apps, they connect to the Internet and send data to a server. The server then retrieves that data, interprets it, performs the necessary actions, and sends it back to your phone. The application then interprets that data and presents you with the information you wanted in a readable way.

Now that we understand what API means, lets get to know a little about FastAPI.

What is FastAPI?
FastAPI is a modern, fast (high-performance), web framework for building APIs with Python 3.7+ based on standard Python type hints.

Some key features of FastAPI include;

Fast to code: Increase the speed to develop features by about 200% to 300%. *
Fewer bugs: Reduce about 40% of human (developer) induced errors. *
Intuitive: Great editor support. Completion everywhere. Less time debugging.
Easy: Designed to be easy to use and learn. Less time reading docs.
Short: Minimize code duplication. Multiple features from each parameter declaration. Fewer bugs.
Robust: Get production-ready code. With automatic interactive documentation.
Standards-based: Based on (and fully compatible with) the open standards for APIs:
You can learn more about FastAPI in the link provided at the end of this article.

First and foremost, I trained machine learning models with the titanic dataset retrieved from kaggle. A link to my source code will be provided at the end of this article. This will show how I cleaned the dataset to make ready for training and evaluation.

The processes I employed in creating my API are stated below;

Export machine learning model to local PC
Install the required packages to be able to run evaluation locally.
Set up virtual environment (venv)
Import required and relevant libraries
Configuration and Setup
Loading the exported assets
Setting your BaseModel
Endpoints.
1. Export machine learning model to local PC
This is the first step I took in processing my API. Exports are taken from my initial Jupyter notebook. A link will be provided as stated earlier to view the source codes. The ML items exported include ; the chosen model, pipeline preprocessor . These various items can be exported individually but for ease of access, I created a dictionary to export the ML Items at a go. Pickle was used in exporting the ML items. I also saved the exported file in the same file location as my api python file.

Below is an illustration of how to export the ML items using dictionary and pickle.


Fig 1.
Fig 1, shows the code used in exporting my model to local PC.

2. Set up virtual environment (venv)
This step involved creating a repository or folder for exported items. A python script was also be needed to host the backend codes for the success of the app. I created a virtual environment to prevent any disputes with any related variables. Below is the code I used in activating my virtual environment and setting up my API


#For Mac/Linux users 
python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt 

#For windows users
python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
3. Import required and relevant libraries
To be able to build and set up your API, you need to import the relevant libraries.

from fastapi import FastAPI
import pickle, uvicorn, os
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.compose import make_column_selector as selector
from sklearn.metrics import accuracy_score
Above are the libraries I used in building my API. Let’s note that, pydantic is primarily a parsing library, not a validation library. Validation is a means to an end: building a model which conforms to the types and constraints provided.

In other words, pydantic guarantees the types and constraints of the output model, not the input data.

Also note that, you have to install the packages that are not available in your requirement file which was also created when setting up the virtual environment.

4. Configuration and Setup
After importation of libraries, we then configure and set up our API. We specify the paths of each variable created in the environment.

## Variables of environment
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "assets")
Ml_api_pipeline = os.path.join(ASSETSDIRPATH, "Ml_api_pipeline.pkl")
NB; assets is the folder which contains my exported machine learning model and pipeline.

The exported ml model and pipeline is name ‘Ml_api_pipeline.pkl’.

# Api Basic Config
app = FastAPI(
    title="Titanic Survivors API",
    version="0.0.1",
    description="Titanic Survivors Prediction",
)
The above code is a basic configuration of the API. The result achieved from running this code is shown in Fig 2. below.


Fig 2.
5. Loading the exported assets
## Loading of assets
with open(Ml_api_pipeline, "rb") as f:
    loaded_items = pickle.load(f)Setting up the backend to process inputs and display outputs
6. Setting up the back end to process inputs and display outputs
This section replicates the steps taken in my initial Jupyter notebook. After loading the required assets and libraries, you then build the backend to make prediction. As stated in the early lines of this section, the steps taken in the Jupyter notebook are replicated, ie,

a. Receiving inputs

b. preprocessing

c. Predicting and returning the output of predictions.

7. Endpoints
Simply put, an endpoint is one end of a communication channel. When an API interacts with another system, in this case, my machine learning models, the touchpoint of this communication are considered endpoints. For APIs, an endpoint can include a URL of a server or service. Each endpoint is the location from which APIs can access the resources they need to carry out their function.

APIs work using ‘requests’ and ‘responses.’ When an API requests information from a web application or web server, it will receive a response. The place that APIs send requests and where the resource lives, is called an endpoint.

@app.post("/Titanic")
async def predict(input: ModelInput):
    """__descr__
    --details---
    """
    output_pred = make_prediction(
        SibSp =input.SibSp,
        Pclass =input.Pclass,
        Age =input.Age,
        Parch =input.Parch,
        Fare =input.Fare,
        Embarked_C= input.Embarked_C,
        Embarked_Q =input.Embarked_Q,
        Sex_female= input.Sex_female
    )

    # Labelling Model output
    if output_pred == 0:
        output_pred = "No,the person didn't survive"
    else:
        output_pred = "Yes,the person survived"
    #return output_pred
    return {
        "prediction": output_pred,
        "input": input
    }

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        reload=True,
    )
Above is a snippet of the codes that constitute my endpoint.


Fig 3.
Fig 3. shows a screenshot of how my API looks after all the processes are complete.

As promised, here is a link to my repository. This contains all source codes. That of my Jupyter notebook and The api python file.

GitHub - SampsonChris/Titanic-Survivors-Project
The purpose of this project is to create a machine learning API endpoint to predict against the inputs received from…
github.com

I hope this read was worth your time ☺️✌🏾.

References

https://smartbear.com/learn/performance-monitoring/api-endpoints/#:~:text=Simply%20put%2C%20an%20endpoint%20is,of%20a%20server%20or%20service.

https://www.mulesoft.com/resources/api/what-is-an-api#:~:text=Many%20people%20ask%20themselves%2C%20%E2%80%9CWhat,data%20within%20and%20across%20organizations.

https://fastapi.tiangolo.com/





