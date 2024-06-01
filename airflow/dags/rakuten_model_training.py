from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.operators.email import EmailOperator
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import timedelta, datetime, timezone
import pickle
import os
import json
import requests
import pandas as pd

# Arguments par défaut
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=1),
}

# Creation du DAG Fine-tuning
with DAG(
    dag_id='rakuten_model_finetuning',
    tags=['rakuten', 'datascientest'],
    default_args=default_args,
    description="Vérifie l'accuracy des 10 derniere prédiction, relance le finetuning si nécéssaire",
    schedule_interval=timedelta(minutes=1),
    start_date=days_ago(1),
    catchup=False
) as dag:
    
    def read_last_execution_time(file_path):
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                last_execution_time_str = f.read()  # Lire le contenu du fichier
                last_execution_time = datetime.fromisoformat(last_execution_time_str) 
        else:
            last_execution_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        return last_execution_time

    def write_last_execution_time(file_path, last_execution_time):
        with open(file_path, 'w') as f:  
            f.write(last_execution_time.isoformat())

    def check_file_modification():
        # Chemin du fichier à surveiller
        file_path = '/app/data/preprocessed/new_classes.csv'
        # Chemin du fichier pour enregistrer la dernière exécution
        execution_time_file_path = '/tmp/last_execution_time.txt'
        
        # Lire le dernier temps d'exécution
        last_execution_time = read_last_execution_time(execution_time_file_path)

        # Vérifie si le fichier existe
        if os.path.exists(file_path):
            # Récupère l'horodatage de la dernière modification du fichier
            file_modified_time = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)

            # Compare les horodatages pour déterminer si le fichier a été modifié
            if file_modified_time > last_execution_time:
                print(f"#### last_execution_time: {last_execution_time}")
                print(f"#### file_modified_time: {file_modified_time}")
                # Mettre à jour le dernier temps d'exécution
                last_execution_time = file_modified_time
                # Enregistrer le nouveau temps d'exécution
                write_last_execution_time(execution_time_file_path, last_execution_time)
                print("#### Le fichier a été modifié depuis la dernière exécution.")
                return 'check_model_accuracy'
            else:
                print("#### Le fichier n'a pas été modifié depuis la dernière exécution.")
                return 'skip_processing'
        else:
            print("#### Le fichier n'existe pas.")
            raise Exception("#### Le fichier n'existe pas.")

        
    check_file_modification_task = BranchPythonOperator(
        task_id='check_file_modification',
        python_callable=check_file_modification,
        )
    
    """   
    file_sensor_task = FileSensor(
        task_id='file_sensor',
        filepath='/app/data/preprocessed/new_classes.csv',  # Chemin vers le fichier new_classes.csv à surveiller
        fs_conn_id="fs_default",  # Connexion FS configurée dans Airflow
        poke_interval=30,  # Intervalle de vérification en secondes
        timeout=600,  # Timeout en secondes
        mode='reschedule'  # Configuration du mode reschedule
    )
    """
    skip_processing = DummyOperator(
        task_id='skip_processing'
    )

    def check_accuracy(**context):
        try:
            api_key = Variable.get('api_key')
            num_sales = int(Variable.get('num_sales', default_var=10))
            new_sales_df = pd.read_csv("/app/data/preprocessed/new_classes.csv")
            num_new_sales = len(new_sales_df)
            if (num_new_sales >= num_sales):
                response = requests.get(
                    'http://api_flows:8003/compute_metrics',
                    headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {api_key}"},
                    data=json.dumps({
                        "classes_path": "data/preprocessed/new_classes.csv",
                        "num_sales": num_sales,
                        "api_secured": True
                        })
                    )
                if response.status_code == 200: 
                    accuracy = response.json().get("accuracy", None)
                    print("#### Accuracy on the last",num_sales,"sales = ",accuracy)
                    context['ti'].xcom_push(key='accuracy', value=accuracy)
                else:
                    print('#### The task check_accuracy did not work!')
                    raise Exception('#### The task check_accuracy did not work!')
            else:
                print(f'#### The number of new sales {num_new_sales} is less than the number of sales expected {num_sales} to compute the accuracy!')
        except requests.RequestException as e:
            print(f"#### Failed to compute metrics : {e}")
        return

    check_task = PythonOperator(
        task_id='check_model_accuracy',
        python_callable=check_accuracy,
        provide_context=True
    )
    
    def train_model(n_epochs=1, **context):       

        # Dossier template pour le réentrainement
        model_dir = "/empty_model"
                        
        try:
            api_key = Variable.get('api_key')
            accuracy = context['ti'].xcom_pull(key='accuracy')
            threshold = float(Variable.get('min_accuracy_threshold', default_var=0.7))
            n_sales_ft = int(Variable.get('n_sales_finetuning', default_var=100))
            if (accuracy < threshold):
                subfolders = [f.name for f in os.scandir('/app/models') if f.is_dir() and "saved_model" in f.name and "Full" in f.name]
                # Trier les sous-dossiers de manière descendante seulement si la liste n'est pas vide
                if subfolders:
                    sorted_subfolders = sorted(subfolders, reverse=True)
                    model_dir = sorted_subfolders[O]  # Selection du dernier entrainement complet
                else:
                    sorted_subfolders = []
                    print('#### Fine-tuning is impossible due to lack of Full traim model')
                    raise Exception('#### The task train_model did not work!')
                    
                response = requests.get(
                    'http://api_flows:8003/save_model_start_train',
                    headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {api_key}"},
                    data=json.dumps({
                        "model_path": "/app/models"+model_dir,
                        "dataset_path":"/app/data/preprocessed",
                        "n_epochs": n_epochs,
                        "samples_per_class":0,
                        "full_train":False,
                        "n_sales_ft": n_sales_ft,
                        "api_secured": True
                        })
                    )
                if response.status_code == 200:
                    print("#### Training processed over !")
                    print("#### "+response.json().get("message", "No message in response"))
                else:
                    print("#### Failed to train the model")
                    print("#### "+response.json().get("message", "No message in response"))
                    raise Exception('#### The task train_model did not work!')
            else:
                print('#### Train is not necessary')
        except requests.RequestException as e:
            print(f"#### Failed to train the model : {e}")
        return
    
    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        op_kwargs={'n_epochs': 30},
        provide_context=True
    )
       
    def send_direct_message(**context):
        slack_api_token = Variable.get('slack_api_token')
        client = WebClient(token=slack_api_token)
        accuracy = context['ti'].xcom_pull(key='accuracy')
        threshold = float(Variable.get('min_accuracy_threshold', default_var=0.7))
        message = f"The accuracy of the model is below the threshold of {threshold}. Current accuracy: {accuracy}"
        slack_email_list = Variable.get('slack_email_list', default_var="['olivier.renouard1103@gmail.com']")
        slack_email_list = eval(slack_email_list)
        if accuracy < threshold:
            for email in slack_email_list:
                try:
                    # Trouver l'ID de l'utilisateur par email
                    response = client.users_lookupByEmail(email=email)
                    user_id = response['user']['id']
                    
                    # Envoyer le message direct
                    response = client.chat_postMessage(
                        channel=user_id,
                        text=message
                    )
                    print(f"#### Slack message sent to {user_id}: {response['message']['text']}")
                except SlackApiError as e:
                    print(f"#### Error sending Slack message: {e.response['error']}")
            
    notify_user_task = PythonOperator(
        task_id='alert_slack_user',
        python_callable=send_direct_message,
        provide_context=True
    )

    def send_email_alert(**context):
        accuracy = context['ti'].xcom_pull(key='accuracy')
        threshold = float(Variable.get('min_accuracy_threshold', default_var=0.7))
        email_list = Variable.get('email_list', default_var="['olivier.renouard1103@gmail.com']")
        email_list = eval(email_list)
        if accuracy < threshold:
            for email in email_list:
                email_op = EmailOperator(
                    task_id='send_email',
                    to=email,
                    subject='Model Accuracy Alert',
                    html_content=f"<p>The accuracy of the model is below the threshold of {threshold}. Current accuracy: {accuracy}</p>"
            )
            email_op.execute(context)
    
    send_email_task = PythonOperator(
        task_id='alert_email',
        python_callable=send_email_alert,
        provide_context=True
    )
    
    # Définition de l'ordre des tâches
    check_file_modification_task >> check_task
    check_file_modification_task >> skip_processing
    # file_sensor_task >> check_task
    check_task >> send_email_task
    check_task >> notify_user_task
    check_task >> train_model_task
    
# Creation du DAG Traing
# Arguments par défaut
default_args2 = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    'timezone': 'Europe/Paris'  # Définir le fuseau horaire sur Paris
}

with DAG(
    dag_id='rakuten_model_training',
    tags=['rakuten', 'datascientest'],
    default_args=default_args2,
    description="Execute l'entrainement quotidien",
    schedule_interval='0 22 * * *',  # Exécuter une fois par jour à 23h,
    start_date=days_ago(1),
    catchup=False
) as dag_train:
    
    def full_train_model(n_epochs=10, **context):       

        # Dossier template pour le réentrainement
        model_dir = "/empty_model"
                        
        try:
            api_key = Variable.get('api_key')               
            response = requests.get(
                'http://api_flows:8003/save_model_start_train',
                headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {api_key}"},
                data=json.dumps({
                    "model_path": "/app/models"+model_dir,
                    "dataset_path":"/app/data/preprocessed",
                    "n_epochs": n_epochs,
                    "samples_per_class":0,
                    "full_train":True,
                    "api_secured": True
                    })
                )
            if response.status_code == 200:
                print("#### Dayly training processed over !")
                print("#### "+response.json().get("message", "No message in response"))
            else:
                print("#### Failed to achieve the dayly train")
                print("#### "+response.json().get("message", "No message in response"))
                raise Exception('#### The dayly task train_model did not work!')
        except requests.RequestException as e:
            print(f"#### Failed to train the model : {e}")
        return
    
    full_train_model_task = PythonOperator(
        task_id='full_dayly_train_model',
        python_callable=full_train_model,
        op_kwargs={'n_epochs': 10},
        provide_context=True
    )
       