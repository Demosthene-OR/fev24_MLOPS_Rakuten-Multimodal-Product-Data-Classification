from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.email import EmailOperator
from airflow.providers.slack.operators.slack import SlackAPIPostOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from datetime import timedelta, datetime
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
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Creation du DAG
with DAG(
    dag_id='check_accuracy',
    tags=['rakuten', 'datascientest'],
    default_args=default_args,
    description="Vérifie l'accuracy des 10 derniere prédiction, relance l'entrainement si nécéssaire",
    schedule_interval=timedelta(minutes=1),
    start_date=days_ago(1),
    catchup=False
) as dag:
    
    def check_accuracy(num_sales = 10, **context):
        try:
            api_key = Variable.get('api_key')
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
                raise Exception('#### The task check_accuracy did not work!')
        except requests.RequestException as e:
            print(f"#### Failed to compute metrics : {e}")
        return

    check_task = PythonOperator(
        task_id='check_model_accuracy',
        python_callable=check_accuracy,
        op_kwargs={'num_sales': 10},
        provide_context=True
    )
    
    def train_model(n_epochs=1, samples_per_class=5, threshold=0.7, **context):       

        # Dossier template pour le réentrainement
        model_dir = "/empty_model"
                        
        try:
            api_key = Variable.get('api_key')
            accuracy = context['ti'].xcom_pull(key='accuracy')
            if (accuracy < threshold):
                response = requests.get(
                    'http://api_flows:8003/save_model_start_train',
                    headers={'Content-Type': 'application/json', 'Authorization': f"Bearer {api_key}"},
                    data=json.dumps({
                        "model_path": "models"+model_dir,
                        "dataset_path":"data/preprocessed",
                        "n_epochs": n_epochs,
                        "samples_per_class":samples_per_class,
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
        op_kwargs={'n_epochs': 1, 'samples_per_class':5, 'threshold':0.7},
        provide_context=True
    )
       
    def send_direct_message(**context):
        slack_api_token = Variable.get('slack_api_token')
        client = WebClient(token=slack_api_token)
        accuracy = context['ti'].xcom_pull(key='accuracy')
        message = f"The accuracy of the model is below the threshold of {threshold}. Current accuracy: {accuracy}"
        threshold = Variable.get('min_accuracy_threshold', default_var=0.7)
        email_list = Variable.get('email_list', default_var="['olivier.renouard1103@gmail.com']")
        email_list = eval(email_list)
        if accuracy < threshold:
            for email in email_list:
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
        threshold = Variable.get('min_accuracy_threshold', default_var=0.7)
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
    check_task >> send_email_task
    check_task >> notify_user_task
    check_task >> train_model_task