# Titanic-Survivors-Project

# Machine Learning API using the Titanic Dataset with FastAPI
The purpose of this project is to create a machine learning API endpoint to predict against the inputs received from the user

# Setup
The commands below will help execute this project

Install the required packages to be able to run the evaluation locally.

You need to have Python 3 on your system (a Python version lower than 3.10). Then you can clone this repo and being at the repo's root :: repository_name> ... follow the steps below:

- Windows:

  python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  
  
- Linux & MacOs:

  python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  \
  
The both long command-lines have a same structure, they pipe multiple commands using the symbol ; but you may manually execute them one after another.


Run FastAPI
Run the demo apps (being at the repository root):

FastAPI:

Demo

uvicorn src.maths.api:app --reload 
