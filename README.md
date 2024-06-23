Project Name
==============================

This project was carried out as part of the **Datascientest MLOps course**. Its aim is to design a platform that predicts the category of a new product for sale on the **Rakuten marketplace**, and to keep the model up to date and efficient.
The models are based on Deep Learning and use the product description and its image.
If you'd like to find out more, read this [presentation]().

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── airflow            <- Airflow service (User: airflow, Pwd: aiflow).
    │   ├── Variables Airflow.json           <- To upload in order to set all variables
    │   ├── dags           <- Contain the dags
    │   ├── logs           
    │   └── plugins            
    ├── data
    │   ├── external       <- Data from third party sources -> the external data you want to make a prediction on
    │   ├── predict        <- Data (input and output) to make a prediction on
    |   ├── mysql-data     <- Rakuten_db in MySQL with users and their rights (Not in the repo. Will be created with docker-compose)
    │   ├── preprocessed   <- The final, canonical data sets for modeling.
    |   |  ├── image_train <- Where you put the images of the train set
    |   |  ├── image_test <- Where you put the images of the predict set
    |   |  ├── X_train_update.csv   <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    |   |  └── X_test_update.csv    <- The csv file with te columns designation, description, productid, imageid like in X_train_update.csv
    │   └── raw            <- The original, immutable data dump.
    |      ├── image_train <- Where you put the images of the train set
    |      └── image_test <- Where you put the images of the predict set
    |
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
    ├── tensorboard        <- Contain the nessecary file to create a docker container
    │
    └── tests              <- Source code of the tests
    │   └── test_rakuten.py   <- Scripts of all the test which are launched by Gihub Actions @ each commit
    │
    └── Rakuten.postman_collection.json     <- This collection contains all the API requests

--------
Using the repository with Docker
------------
Once you've downloaded the github folder, the easiest way to use it is with **Docker**, which lets you access the user database and use the APIs securely.
  
> `./docker/setup.sh`                    <- To do in Git bash. It will run the process to build and launch the containers with all the API    

Two containers, to manage the user database, will be launched: **mysql** & **adminer**. The database is located in the folder *data/mysql-data*  
Then, in your browser you can launch the database adminer which show you the registered users:  
http://localhost:8080/?server=users_db&username=root&db=rakuten_db&select=Users  
password = Rakuten  

We suggest to use **[Postman](https://www.postman.com/)** to run the API. If the case, you can upload all the available requests with the file [Rakuten.postman_collection.json](https://github.com/Demosthene-OR/fev24_MLOPS_Rakuten-Multimodal-Product-Data-Classification/blob/main/Rakuten.postman_collection.json) in the root folder  

> However, without Postman, you can then run:  
    **1** - Token generation (to login) :  
>   `curl 'http://localhost:8001/token' --header 'Content-Type: application/x-www-form-urlencoded' \`    
>       `--data-urlencode 'username=John' --data-urlencode 'password=John'`    
    **2** - Check the informations about the user designated by the token :   
>   `curl 'http://localhost:8001/secured' --header 'Authorization: Bearer "Obtained access token"'`  
    **3** - Predict  
        The input data are located, by default, in data/predict.  (There are many paramteters available)   
        The predictions are saved in data/predict as 'predictions.csv'    
>   `curl 'http://localhost:8000/prediction' --header 'Authorization: Bearer "Obtained access token"' \`  
>        `--header 'Content-Type: application/json' --data '{"api_secured": "True"}'`  
    **4** - Train   
>   `curl 'http://localhost:8002/train' --header 'Content-Type: application/json' --header 'Authorization: Bearer "Obtained access token"' --data '{"api_secured": "True"}'`  
        There are many parameters available, and many other endpoints to discover.  

If you want to visualize and track various aspects of the machine learning models (Text & Image) during training, use **tensorboard**:  
    Go on you browser and enter "http://localhost:6006"

You can also use **Airflow**, to update automatically the models when the performances are below a certain threshold.  
**Caution**: Upload the file 'Variables Airflow.json' in the folder airflow:  
http://localhost:8501/

Finally, if you want to have an overview of the full process, we encourage you to launch **Streamlit** which will go through all the proccess, and will give you access to the user database, tensorflow and Airflow:  
http://localhost:8501/`  

**Note**: All the APIs are automatically tested on Github push & pull request

--------
Using the repository without **Docker**
------------
Open the anaconda powershell on the root of the project and follow those instructions :

> `conda create -n "Rakuten-project"`    <- It will create your conda environement

> `conda activate Rakuten-project`       <- It will activate your environment

> `conda install pip`                    <- May be optionnal

> `pip install -r requirements.txt`      <- It will install the required packages. Caution: You must install python 3.10.14 before this

> `python src/data/import_raw_data.py`   <- It will import the tabular data on data/raw/

> `python src/main.py`                   <- It will train the models on the dataset and save them in models. By default, the number of epochs = 1

> `python src/predict.py`                <- It will use the trained models to make a prediction (of the prdtypecode) on the desired data, by default it will predict on the train. You can pass the path to data and images as arguments if you want to change it  
Exemple :  

> `python src/predict.py --dataset_path "data/preprocessed/X_test_update.csv" --images_path "data/preprocessed/image_test"`                           
> The predictions are saved in data/preprocessed as 'predictions.csv'  

> If you want with to predict with the API in a unsecured way (without Docker) :  
> `uvicorn src.predict_API:app --reload`  
> `curl 'http://localhost:8000/prediction' --header 'Authorization: Bearer' --header 'Content-Type: application/json' --data '{}'`  

> If you want with to train the model with the API in a unsecured way (without Docker) :   
> `uvicorn src.main_API:app --port 8002 --reload`  
> `curl 'http://localhost:8002/train' --header 'Content-Type: application/json' --header 'Authorization: Bearer ' --data '{}'`  

If you want to visualize and track various aspects of the machine learning models (Text & Image) during training, launch **tensorboard**:  
> `tensorboard --logdir=logs/`  
    Then go on you browser and enter "http://localhost:6006"  
  
If you want to run the API in a secured way (with Docker), stop uvicorn first   


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

