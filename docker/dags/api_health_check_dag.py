from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import requests
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
    'api_health_check',
    default_args=default_args,
    description='DAG to check API health and notify if down',
    schedule_interval='@hourly',  # Change as needed  @daily, @weekly, */30 * * * * pour toutes les 30 minutes
)

# Fonction pour vérifier l'état de santé de l'API
def check_api_health(**kwargs):
    check_url = Variable.get("api_check_url")
    try:
        response = requests.get(check_url)
        if response.status_code != 200:
            notify_team(f"API Health Check Failed", f"API returned status code {response.status_code}")
    except requests.RequestException as e:
        notify_team(f"API Health Check Failed", f"API request failed with exception: {e}")

# Fonction pour notifier l'équipe par email
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

# Définir la tâche de vérification de l'API
check_api_health_task = PythonOperator(
    task_id='check_api_health',
    provide_context=True,
    python_callable=check_api_health,
    dag=dag,
)

check_api_health_task
