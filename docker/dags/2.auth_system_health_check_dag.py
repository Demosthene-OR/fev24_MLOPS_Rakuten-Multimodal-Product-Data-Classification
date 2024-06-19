import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import requests
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from airflow.models import Variable
from airflow.operators.email import EmailOperator
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
    api_url = Variable.get(key="api_oauth_url")
    test_first_name = Variable.get(key="test_first_name")
    test_last_name = Variable.get(key="test_last_name")
    test_email = Variable.get(key="test_email")
    test_authorization = Variable.get(key="test_authorization")
    test_username = Variable.get(key="test_username")
    test_password = Variable.get(key="test_password")
    
    # Format the user information into a single string
    user_info = f"FirstName: {test_first_name}, LastName: {test_last_name}, Email: {test_email}, Authorization: {test_authorization}, username: {test_username}, password: {test_password}"

    # Log the formatted user information
    logging.info(user_info)
    
    logging.info(api_url)
    
    url = api_url + 'usercreate'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
    }

    data = {
        "FirstName": test_first_name,
        "LastName": test_last_name,
        "Email": test_email,
        "Authorization": test_authorization,
        "username": test_username,
        "password": test_password
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()  # Raise an error for bad status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Failed to add user: {e}")
    
    
# Fonction pour vérifier l'authentification
def check_auth_system(**kwargs):
    ti = kwargs['ti'] # TaskInstance to push XCom variables
    # URLs and credentials from Airflow variables
    auth_url = Variable.get(key="api_oauth_url") + 'token'
    test_username = Variable.get("test_username")
    test_password = Variable.get("test_password")
    wrong_username = Variable.get("test_wrong_username")
    wrong_password = Variable.get("test_wrong_password")
    
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    
    def authenticate(username, password):
        data = {
            'grant_type': 'password',
            'username': username,
            'password': password,
            'scope': '',
            'client_id': 'string',
            'client_secret': 'string'
        }
        
        try:
            response = requests.post(auth_url, headers=headers, data=data)
            response.raise_for_status()
            logging.info(f"Authentication successful for user: {username}")
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logging.error(f"HTTP error occurred for user {username}: {http_err}")
        except Exception as err:
            logging.error(f"Error occurred for user {username}: {err}")
        
        return None

    # Authenticate with correct credentials
    correct_auth_response = authenticate(test_username, test_password)
    
    if correct_auth_response:
        logging.info("Correct credentials authentication response: %s", correct_auth_response)
        # Push the access token to XCom
        ti.xcom_push(key='access_token', value=correct_auth_response.get('access_token'))
    else:
        logging.error("Failed to authenticate with correct credentials")
    
    # Authenticate with incorrect credentials
    incorrect_auth_response = authenticate(wrong_username, wrong_password)
    
    if incorrect_auth_response:
        logging.error("Authentication unexpectedly succeeded with incorrect credentials: %s", incorrect_auth_response)
    else:
        logging.info("Authentication correctly failed with incorrect credentials")      

# Fonction pour vérifier l'autorisation
def check_authorization(**kwargs):
    
    ti = kwargs['ti'] # TaskInstance to push XCom variables
    url =  Variable.get(key="api_oauth_url") + Variable.get(key="auth_secured_endpoint")

    headers = {
        'accept': 'application/json'
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        logging.info("Successful Response")
        # Push the authorization to XCom
        ti.xcom_push(key='authorization', value=response.json()['detail'])
        logging.info(response.json())
        # notify_team("Auth System Check Success", f"Access to protected resource succeded with status code {response.status_code}")
    elif response.status_code == 401:
        logging.info("Error: Unauthorized")
        # Push the authorization to XCom
        ti.xcom_push(key='authorization', value=response.json()['detail'])
        logging.info(response.json())
        # notify_team("Auth System Check Failed", f"Access to protected resource failed with status code {response.status_code}")
    else:
        logging.info(f"Unexpected status code: {response.status_code}")
        logging.info(response.text)
        # Push the authorization to XCom
        ti.xcom_push(key='authorization', value=response.text) 
        # notify_team("Auth System Check Failed", f"Access to protected resource request failed with exception: {e}")

# Fonction pour supprimer un utilisateur
def delete_user(username, **kwargs):
    user_deletion_url = Variable.get("api_oauth_url") + "userdelete/" + username

    headers = {
        'accept': '*/*'
    }
    data = ''
    
    try:
        response = requests.post(user_deletion_url, headers=headers, data=data)
        if response.status_code != 204:
            logging.info("User Deletion Failed", f"Failed to delete test user with status code {response.status_code}")
    except requests.RequestException as e:
        logging.info("User Deletion Failed", f"User deletion request failed with exception: {e}")

def generate_email_content(**kwargs):
    ti = kwargs['ti']
    tasks_statuses = {
        'Add User Task': ti.xcom_pull(task_ids='add_user'),
        'Check Auth Task': ti.xcom_pull(task_ids='check_auth_system'),
        'Check Authorization Task': ti.xcom_pull(task_ids='check_authorization'),
        'Delete User Task': ti.xcom_pull(task_ids='delete_user'),
    }
    
    email_content = f"""
    <html>
        <body>
            <h2>DAG Execution Summary</h2>
            <ul>
                <li><strong>Add User Task:</strong> {tasks_statuses['Add User Task']}</li>
                <li><strong>Check Auth Task:</strong> {tasks_statuses['Check Auth Task']}</li>
                <li><strong>Check Authorization Task:</strong> {tasks_statuses['Check Authorization Task']}</li>
                <li><strong>Delete User Task:</strong> {tasks_statuses['Delete User Task']}</li>
            </ul>
            <p>For more details, please check the Airflow UI.</p>
        </body>
    </html>
    """
    
    return email_content

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
    op_kwargs= {
        'username': Variable.get(key="test_username")
    },
    dag=dag,
)

# Définition de la tâche EmailOperator
send_email_task = EmailOperator(
    task_id='send_email',
    to= Variable.get(key='notification_email'),
    subject='User Management DAG Execution Report for {{ ds }}',
    html_content=f"""
    <html>
        <body>
            <h2>DAG Execution Summary</h2>
            <ul>
                <li><strong>Add User Task:</strong> OK </li>
                <li><strong>Check Auth Task:</strong> OK </li>
                <li><strong>Check Authorization Task:</strong> OK </li>
                <li><strong>Delete User Task:</strong> OK </li>
            </ul>
            <p>For more details, please check the Airflow UI.</p>
        </body>
    </html>
    """,
    dag=dag,
)


# Ordre des tâches
add_user_task >> check_auth_task >> check_authorization_task >> delete_user_task >> send_email_task
#>> email_report
