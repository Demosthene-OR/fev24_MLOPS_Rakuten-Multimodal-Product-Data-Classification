from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import datetime

def print_date():
    print(datetime.datetime.now())

with DAG(
    dag_id='minutely_dag_without_catchup',
    description='My DAG that\'s triggered every minute',
    tags=['tutorial', 'datascientest'],
    schedule_interval='* * * * *',
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
) as my_dag:

    my_task = PythonOperator(
        task_id='print_date_task',
        python_callable=print_date
    )