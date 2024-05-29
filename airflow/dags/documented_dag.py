from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python_operator import PythonOperator
import time

def sleep_1_sec():
        time.sleep(1)

with DAG(
    dag_id="documented_dag",
    tags=['tutorial', 'datascientest'],
    doc_md="""# Documented DAG
This `DAG` is documented and the next line is a quote:

> Airflow is nice

This DAG has been made:

* by DataScientest
* with documentation
* with caution
    """,
    start_date=days_ago(0),
    schedule_interval=None
) as dag:

    task1 = PythonOperator(
        task_id="sleep1",
        python_callable=sleep_1_sec,
        doc_md="""# Task1

    Task that is used to sleep for 1 sec""",
    )

    task2 = PythonOperator(
        task_id="sleep2",
        python_callable=sleep_1_sec,
        doc="""Task 3

    It has an ugly description.
    """
    )
    task1 >> task2