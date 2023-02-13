import gradio as gr
import pandas as pd
import os, pickle
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# Defining a function to load exported ML toolkit                    
def load_saved_objects(filepath='Ml_api_pipeline.pkl'):
    "Function to load saved objects"

    with open(filepath, 'rb') as file:
        loaded_object = pickle.load(file)
    
    return loaded_object

# Loading the toolkit
loaded_toolkit = load_saved_objects('/Users/Admin/Downloads/titanic/Career_Accelerator_P6-ML_API/src/demo_01/assets/Ml_api_pipeline.pkl')

# Instantiating the elements of the Machine Learning Toolkit
print('Instantiating')
preprocessor = loaded_toolkit["pipeline_for_preprocessing"]
num_cols = loaded_toolkit['numerical_columns']
cat_cols = loaded_toolkit['categorical_columns']
best_grid_rf_model = loaded_toolkit['model']

# Relevant data inputs
expected_inputs = ['SibSp','Pclass','Age','Parch','Fare','Embarked','Sex']

#defining the predict function
def predict(*args, model = best_grid_rf_model, pipeline = preprocessor):
    
    # Creating a dataframe of inputs
    Input_data = pd.DataFrame([args], columns= expected_inputs)
    
# Modeling
    model_output = model.predict(Input_data)
    return float(model_output[0])
    #output_str = "Hey there.....👋 your customer will"
    #return(output_str,model_output)



# Working on inputs 
with gr.Blocks() as demo:

        gr.Markdown("# Titanic Survivors Prediction")
        
        #gr.Markdown("Number of Siblings and Parents Onboard")
        with gr.Row():
            SibSp = gr.Dropdown(['0', '1'],label = 'Siblings', value= '0')
            Parch = gr.Dropdown(['0', '1'],label = 'Parents/Children', value= '0')
            Embarked = gr.Radio(['C', 'Q', 'S'],label = 'Port of Embarkation', value= '0')
            #Embarked_Q = gr.Dropdown(['0', '1'],label = 'Port of Embarkation', value= '0')
            Sex = gr.Dropdown(['male', 'female'],label = 'Gender', value= 'male')
            

        #gr.Markdown("Payment Info")
        with gr.Column():
            Age = gr.Slider(0, 100,label = 'Age')
            Fare = gr.Slider(0, 1000,label = 'Passenger fare')
            

        #gr.Markdown("Demography Info")
        with gr.Row():
            Pclass = gr.Radio(['1', '2', '3'],label = 'Passenger Class')

        with gr.Row():
            btn = gr.Button("Predict").style(full_width=True)
            output = gr.Textbox(label="Prediction") 
               
        btn.click(fn=predict,inputs=[SibSp,Pclass,Age,Parch,Fare,Embarked,Sex],outputs=output)

demo.launch(share= True, debug= True)     