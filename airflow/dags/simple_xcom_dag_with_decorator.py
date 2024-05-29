from airflow import DAG
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from airflow.operators.python import get_current_context

import random

@task
def function_with_return():
    return random.uniform(a=0, b=1)

@task
def function_with_return_and_push():
    task_instance = get_current_context()['task_instance']
    value = random.uniform(a=0, b=1)
    task_instance.xcom_push(key="my_xcom_value", value=value)
    return value

@task
def read_data_from_xcom(my_xcom_value):
    print(my_xcom_value)

@dag(
    dag_id='simple_xcom_dag_with_decorator',
    tags=['tutorial', 'datascientest'],
    schedule_interval=None,
    start_date=days_ago(0)
)
def my_dag():

    my_task1 = function_with_return_and_push()
    my_task2 = read_data_from_xcom(my_task1)

my_dag = my_dag()