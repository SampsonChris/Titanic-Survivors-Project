#Imports
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


# Configuration and setup 
## Variables of environment
DIRPATH = os.path.dirname(__file__)
ASSETSDIRPATH = os.path.join(DIRPATH, "assets")
Ml_api_pipeline = os.path.join(ASSETSDIRPATH, "Ml_api_pipeline.pkl")

print(
    f" {'*'*10} Config {'*'*10}\n INFO: DIRPATH = {DIRPATH} \n INFO: ASSETSDIRPATH = {ASSETSDIRPATH} "
)


# Api Basic Config
app = FastAPI(
    title="Titanic Survivors API",
    version="0.0.1",
    description="Titanic Survivors Prediction",
)


## Loading of assets
with open(Ml_api_pipeline, "rb") as f:
    loaded_items = pickle.load(f)
#print("INFO:    Loaded assets:", loaded_items)

preprocessor = loaded_items["pipeline_for_preprocessing"]
num_cols = loaded_items['numerical_columns']
cat_cols = loaded_items['categorical_columns']
best_grid_rf_model = loaded_items['model']


## BaseModel
class ModelInput(BaseModel):
    SibSp: int
    Pclass: int
    Age: float
    Parch: int
    Fare: float
    Embarked_C: float
    Embarked_Q: float
    Sex_female: float

    

def processing_FE(dataset,  imputer=None, FE=None ): 
    if imputer is not None:
        output_dataset = imputer.transform(dataset)
    else:
        output_dataset = dataset.copy()
    if FE is not None:
        output_dataset = FE.transform(output_dataset)

    return output_dataset


def make_prediction(
    SibSp,Pclass,Age,Parch,Fare,Embarked_C,Embarked_Q,Sex_female
    ):
    df = pd.DataFrame([[SibSp,Pclass,Age,Parch,Fare,Embarked_C,Embarked_Q,Sex_female]],
    columns = [SibSp,Pclass,Age,Parch,Fare,Embarked_C,Embarked_Q,Sex_female],
    )
    X = processing_FE(dataset = df, FE = None)
    model_output = best_grid_rf_model.predict(X).tolist()

    return model_output

# def make_prediction(
#      Age,Pclass,Parch,SibSp,Fare,Sex_female,Embarked_C,Embarked_Q
    
# ):

#     df = pd.DataFrame(
#         [
#             [
                
#                 Age,
#                 Pclass,
#                 Parch,
#                 SibSp,
#                 Fare,
#                 Sex_female,
#                 Embarked_C,
#                 Embarked_Q
                
#             ]
#         ],
#         columns=num_cols + cat_cols,
        
#     )
#     print(num_cols + cat_cols)
#     print( [
                
#                 Age,
#                 Pclass,
#                 Parch,
#                 SibSp,
#                 Fare,
#                 Sex_female,
#                 Embarked_C,
#                 Embarked_Q
                
#             ])
        
#     X = df
#     #df[cat_cols] = df[cat_cols].astype("object")
#     output = best_grid_rf_model.predict(X).tolist()
#     return output
    

## Endpoints
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
