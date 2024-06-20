Project Name
==============================

This project is a starting Pack for MLOps projects based on the subject "movie_recommandation". It's not perfect so feel free to make some modifications on it.

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources -> the external data you want to make a prediction on
    │   ├── predict        <- Data (input and output) to make a prediction on
        ├── mysql-data     <- Rakuten_db in MySQL with users and their rights
    │   ├── preprocessed   <- The final, canonical data sets for modeling.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  ├── image_test <- Where you put the images of the predict set
    |   |  ├── X_train_update.csv   <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    |   |  └── X_test_update.csv    <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    │   └── raw            <- The original, immutable data dump.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  └── image_test <- Where you put the images of the predict set
    ├── docker             <- Files to launch Docker-compose and run a MySQL database with users, authorisation process & the predict function
    │
    ├── logs               <- Logs from training and predicting
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── main.py        <- Scripts to train models 
    │   ├── predict.py     <- Scripts to use trained models to make prediction on the files put in ../data/preprocessed
    │   ├── fastapi_oauth  <- Source code for Open Authorization API. It will check if User is recorded in Rakuten_db and the rights he is granted
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   ├── check_structure.py    
    │   │   ├── import_raw_data.py 
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models                
    │   │   └── train_model.py
    │   └── config         <- Describe the parameters used in train_model.py and predict_model.py
    │
    ├── streamlit_app      <- Source code of the streamlit app which shows how to use all the APIs
    │
    └── tests              <- Source code of the tests
    │   └── test_rakuten.py     <- Scripts of all the test which are launched by Gihub Actions @ each commit
    │
    └── Rakuten.postman_collection.json     <- This collection contains all the API requests for the Datascientest Rakuten project

--------

Once you have downloaded the github repo, open the anaconda powershell on the root of the project and follow those instructions :

> `conda create -n "Rakuten-project"`    <- It will create your conda environement

> `conda activate Rakuten-project`       <- It will activate your environment

> `conda install pip`                    <- May be optionnal

> `pip install -r requirements.txt`      <- It will install the required packages. Caution: You must install python 3.10.14 before this

> `python src/data/import_raw_data.py`   <- It will import the tabular data on data/raw/

> Upload the image data folder set directly on local from https://challengedata.ens.fr/participants/challenges/35/, you should save the folders image_train and image_test respecting the following structure

    ├── data
    │   └── raw           
    |   |  ├── image_train 
    |   |  └── image_test 

> `python src/data/make_dataset.py data/raw data/preprocessed`      <- It will copy the raw dataset and paste it on data/preprocessed/

> `python src/main.py`                   <- It will train the models on the dataset and save them in models. By default, the number of epochs = 1

> `tensorboard --logdir=logs/`           <- It will launch tensorborad and you will be able to visualize and track various aspects of the machine learning model during training and evaluation.
                                            In order to see them, you have to go on you browser and run "http://localhost:6006"

> `python src/predict.py`                <- It will use the trained models to make a prediction (of the prdtypecode) on the desired data, by default it will predict on the train. You can pass the path to data and images as arguments if you want to change it  
> Exemple : 
> `python src/predict.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"`                           
> The predictions are saved in data/preprocessed as 'predictions.csv'  

> If you want with to predict with the API in a unsecured way (without Docker) :  
> `uvicorn src.predict_API:app --reload`  
> `curl 'http://localhost:8000/prediction' --header 'Authorization: Bearer' --header 'Content-Type: application/json' --data '{}'`  

> If you want with to train the model with the API in a unsecured way (without Docker) :   
> `uvicorn src.main_API:app --port 8002 --reload`  
> `curl 'http://localhost:8002/train' --header 'Content-Type: application/json' --header 'Authorization: Bearer ' --data '{}'`  
 

> If you want to run the API in a secured way, you have to use Docker (stop uvicorn first):   
> `./docker/setup.sh`                    <- To do in Git bash. It will run the process to build and launch the containers with all the API    

> 2 containers to manage the user database will be launched: mysql & adminer. The database is located in the folder data/mysql-data  
> Then, in your browser you can launch the database adminer which show you the registered users:  
> http://localhost:8080/?server=users_db&username=root&db=rakuten_db&select=Users  
> password = Rakuten  

> You can then run:
    1 - Token generation (to login) :  
>   `curl 'http://localhost:8001/token' --header 'Content-Type: application/x-www-form-urlencoded' \`    
>       `--data-urlencode 'username=John' --data-urlencode 'password=John'`    
    2 - Check the informations about the user designated by the token :   
>   `curl 'http://localhost:8001/secured' --header 'Authorization: Bearer "Obtained access token"'`
    3 - Predict  
        The input data are located, by default, in data/predict.  (There are many paramteters available)   
        The predictions are saved in data/predict as 'predictions.csv'    
>   `curl 'http://localhost:8000/prediction' --header 'Authorization: Bearer "Obtained access token"' \`  
>        `--header 'Content-Type: application/json' --data '{"api_secured": "True"}'`  
    4 - Train   
>   `curl 'http://localhost:8002/train' --header 'Content-Type: application/json' --header 'Authorization: Bearer "Obtained access token"' --data '{"api_secured": "True"}'`  
        There are many paramteters available   
>          
> Finally, if you want to have an overview of the full process, we encourage you to launch Streamlit:
>   `http://localhost:8501/`

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

