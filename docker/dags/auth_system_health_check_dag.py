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
    'auth_system_health_check',
    default_args=default_args,
    description='DAG to check authentication and authorization systems',
    schedule_interval='@hourly',  # Change as needed
)

# Fonction pour ajouter un utilisateur
def add_user(**kwargs):
    user_creation_url = Variable.get("user_creation_url")
    test_first_name = Variable.get("test_first_name")
    test_last_name = Variable.get("test_last_name")
    test_email = Variable.get("test_email")
    test_authorization = Variable.get("test_authorization")
    test_username = Variable.get("test_username")
    test_password = Variable.get("test_password")

    try:
        response = requests.post(user_creation_url, json={
            'FirstName': test_first_name,
            'LastName': test_last_name,
            'Email': test_email,
            'Authorization': test_authorization,
            'username': test_username,
            'password': test_password
        })
        if response.status_code != 201:
            notify_team("User Creation Failed", f"Failed to create test user with status code {response.status_code}")
    except requests.RequestException as e:
        notify_team("User Creation Failed", f"User creation request failed with exception: {e}")

# Fonction pour vérifier l'authentification
def check_auth_system(**kwargs):
    auth_url = Variable.get("auth_url")
    test_username = Variable.get("test_username")
    test_password = Variable.get("test_password")
    wrong_username = Variable.get("wrong_username")
    wrong_password = Variable.get("wrong_password")

    # Vérifier l'authentification avec les bonnes informations d'identification
    try:
        response = requests.post(auth_url, data={'username': test_username, 'password': test_password})
        if response.status_code != 200:
            notify_team("Auth System Check Failed", f"Authentication with correct credentials failed with status code {response.status_code}")
    except requests.RequestException as e:
        notify_team("Auth System Check Failed", f"Authentication request failed with exception: {e}")

    # Vérifier l'authentification avec des informations d'identification incorrectes
    try:
        response = requests.post(auth_url, data={'username': wrong_username, 'password': wrong_password})
        if response.status_code == 200:
            notify_team("Auth System Check Failed", "Authentication with incorrect credentials succeeded unexpectedly")
    except requests.RequestException as e:
        notify_team("Auth System Check Failed", f"Authentication request failed with exception: {e}")

# Fonction pour vérifier l'autorisation
def check_authorization(**kwargs):
    protected_url = Variable.get("protected_url")
    test_username = Variable.get("test_username")
    test_password = Variable.get("test_password")

    # Obtenir un token d'authentification
    auth_url = Variable.get("auth_url")
    try:
        response = requests.post(auth_url, data={'username': test_username, 'password': test_password})
        token = response.json().get("access_token")
    except requests.RequestException as e:
        notify_team("Auth System Check Failed", f"Authorization request failed with exception: {e}")
        return

    # Vérifier l'accès à une ressource protégée
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(protected_url, headers=headers)
        if response.status_code != 200:
            notify_team("Auth System Check Failed", f"Access to protected resource failed with status code {response.status_code}")
    except requests.RequestException as e:
        notify_team("Auth System Check Failed", f"Access to protected resource request failed with exception: {e}")

# Fonction pour supprimer un utilisateur
def delete_user(**kwargs):
    user_deletion_url = Variable.get("user_deletion_url")
    test_username = Variable.get("test_username")

    try:
        response = requests.delete(user_deletion_url, json={'username': test_username})
        if response.status_code != 204:
            notify_team("User Deletion Failed", f"Failed to delete test user with status code {response.status_code}")
    except requests.RequestException as e:
        notify_team("User Deletion Failed", f"User deletion request failed with exception: {e}")

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

# Définir les tâches de vérification de l'authentification et de l'autorisation
add_user_task = PythonOperator(
    task_id='add_user',
    provide_context=True,
    python_callable=add_user,
    dag=dag,
)

check_auth_task = PythonOperator(
    task_id='check_auth_system',
    provide_context=True,
    python_callable=check_auth_system,
    dag=dag,
)

check_authorization_task = PythonOperator(
    task_id='check_authorization',
    provide_context=True,
    python_callable=check_authorization,
    dag=dag,
)

delete_user_task = PythonOperator(
    task_id='delete_user',
    provide_context=True,
    python_callable=delete_user,
    dag=dag,
)

# Ordre des tâches
add_user_task >> check_auth_task >> check_authorization_task >> delete_user_task
