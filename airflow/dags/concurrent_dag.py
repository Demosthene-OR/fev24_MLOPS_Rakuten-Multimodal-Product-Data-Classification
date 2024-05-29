from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import time


def wait_1_minute():
    time.sleep(60)


with DAG(
    dag_id="concurrent_dag",
    tags=['tutorial', 'datascientest'],
    schedule_interval=None,
    start_date=days_ago(0),
    # default_args={
    #     'pool': 'micro_pool'
    # }
) as dag:

    task1 = PythonOperator(
        task_id="wait1",
        python_callable=wait_1_minute,
    )

    task2 = PythonOperator(
        task_id="wait2",
        python_callable=wait_1_minute,
    )

    task3 = PythonOperator(
        task_id="wait3",
        python_callable=wait_1_minute,
    )