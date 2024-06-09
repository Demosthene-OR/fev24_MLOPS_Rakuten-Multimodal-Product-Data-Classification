from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import requests
import mlflow
import mlflow.sklearn
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from airflow.models import Variable

# Configuration
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
}

dag = DAG(
    'product_category_prediction',
    default_args=default_args,
    description='DAG to automate product category prediction and save results',
    schedule_interval='@daily',  # Change as needed
)

# Fonction pour la préparation des données
def prepare_data(**kwargs):
    data_path = Variable.get("data_path")
    images_path = Variable.get("images_path")
    prediction_output_path = Variable.get("prediction_output_path")
    
    # Lire et traiter les données si nécessaire
    data = pd.read_csv(data_path)
    
    # Enregistrer les données préparées (optionnel)
    data.to_csv(prediction_output_path + "/prepared_data.csv", index=False)

# Fonction pour faire une prédiction
def make_prediction(**kwargs):
    prediction_endpoint = Variable.get("prediction_endpoint")
    data_path = Variable.get("data_path")
    images_path = Variable.get("images_path")
    prediction_output_path = Variable.get("prediction_output_path")
    
    input_data = {
        "dataset_path": data_path,
        "images_path": images_path,
        "prediction_path": prediction_output_path,
        "api_secured": False  # or True if you need to use OAuth
    }

    response = requests.post(prediction_endpoint, json=input_data)
    if response.status_code != 200:
        notify_team("Prediction Failed", f"Prediction request failed with status code {response.status_code}")
    else:
        return response.json()

# Fonction pour le suivi des prédictions avec MLflow
def log_predictions_to_mlflow(**kwargs):
    mlflow_tracking_uri = Variable.get("mlflow_tracking_uri")
    mlflow_experiment_name = Variable.get("mlflow_experiment_name")
    prediction_output_path = Variable.get("prediction_output_path")
    
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment_name)
    
    with mlflow.start_run():
        predictions = pd.read_csv(prediction_output_path + "/predictions.csv")
        
        for index, row in predictions.iterrows():
            mlflow.log_metric("prediction", row['cat_pred'])
        
        mlflow.log_artifact(prediction_output_path + "/predictions.csv")

# Fonction pour sauvegarder les résultats
def save_results(**kwargs):
    prediction_output_path = Variable.get("prediction_output_path")
    
    predictions = pd.read_csv(prediction_output_path + "/predictions.csv")
    predictions.to_csv(prediction_output_path + "/final_predictions.csv", index=False)

# Fonction pour notifier l'équipe en cas d'erreur
def notify_team(subject, message):
    smtp_server = Variable.get("smtp_server")
    smtp_port = Variable.get("smtp_port")
    smtp_user = Variable.get("smtp_user")
    smtp_password = Variable.get("smtp_password")
    notification_email = Variable.get("notification_email")
    
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = notification_email
    msg['Subject'] = subject

    msg.attach(MIMEText(message, 'plain'))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        text = msg.as_string()
        server.sendmail(smtp_user, notification_email, text)
        server.quit()
    except Exception as e:
        print(f"Failed to send notification email: {e}")

# Définir les tâches du DAG
prepare_data_task = PythonOperator(
    task_id='prepare_data',
    provide_context=True,
    python_callable=prepare_data,
    dag=dag,
)

make_prediction_task = PythonOperator(
    task_id='make_prediction',
    provide_context=True,
    python_callable=make_prediction,
    dag=dag,
)

log_predictions_to_mlflow_task = PythonOperator(
    task_id='log_predictions_to_mlflow',
    provide_context=True,
    python_callable=log_predictions_to_mlflow,
    dag=dag,
)

save_results_task = PythonOperator(
    task_id='save_results',
    provide_context=True,
    python_callable=save_results,
    dag=dag,
)

# Ordre des tâches
prepare_data_task >> make_prediction_task >> log_predictions_to_mlflow_task >> save_results_task
