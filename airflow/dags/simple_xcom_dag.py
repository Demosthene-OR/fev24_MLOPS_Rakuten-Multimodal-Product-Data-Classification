from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import random

from airflow.models import Variable
my_variable_value = Variable.get(key="my_variable")

"""
def function_with_return(task_instance):
    task_instance.xcom_push(
        key="my_xcom_value",
        value=random.uniform(a=0, b=1)
    )
"""

def function_with_return(task_instance):
    task_instance.xcom_push(
        key="my_xcom_value",
        value={
            "hello": "world",
            "bonjour": "le monde"
        }
    )

def read_data_from_xcom(task_instance):
    print(
        task_instance.xcom_pull(
            key="my_xcom_value",
            task_ids=['python_task']
        )
    )

with DAG(
    dag_id='simple_xcom_dag',
    schedule_interval=None,
    tags=['tutorial', 'datascientest'],
    start_date=days_ago(0)
) as dag:

    my_task = PythonOperator(
        task_id='python_task',
        python_callable=function_with_return
    )
    



    my_task2 = PythonOperator(
    task_id="read_Xcom_value",
    python_callable=read_data_from_xcom
    )

    my_task >> my_task2