from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago


def create_sub_dag(dag_id, schedule_interval, start_date):
    my_sub_dag = DAG(
        dag_id=dag_id,
        schedule_interval=schedule_interval,
        tags=['tutorial', 'datascientest'],
        default_args={
            'start_date': days_ago(0)
        }
    )

    task1 = BashOperator(
        bash_command="echo subdag task 1",
        task_id="my_sub_dag_task1",
        dag=my_sub_dag
    )

    task2 = BashOperator(
        bash_command="echo subdag task 2",
        task_id="my_sub_dag_task2",
        dag=my_sub_dag
    )

    task1 >> task2

    return my_sub_dag