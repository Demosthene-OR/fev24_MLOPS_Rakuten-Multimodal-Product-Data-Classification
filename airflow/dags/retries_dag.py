from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator
import datetime

def failed_task():
    raise Exception('This task did not work!')

with DAG(
    dag_id='retries_dag',
    description='My DAG that will try but fail',
    tags=['tutorial', 'datascientest'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, minute=1),
    },
    catchup=False
) as my_dag:

    task1 = PythonOperator(
        task_id="my_failed_task",
        python_callable=failed_task,
        retries=3,
        retry_delay=datetime.timedelta(seconds=30),
        email_on_retry=True,
        email=['olivier.airflow@gmail.com']
    )