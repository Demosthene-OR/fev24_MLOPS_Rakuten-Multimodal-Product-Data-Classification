from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.subdag import SubDagOperator
from airflow.operators.bash import BashOperator
# importing DAG generating function
from my_subdag import create_sub_dag

my_parent_dag = DAG(
    dag_id="parent_to_subdag",
    schedule_interval=None,
    tags=['tutorial', 'datascientest'],
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, 1)
    }
)

task1 = SubDagOperator(
    task_id="my_subdag",
    subdag=create_sub_dag(
        dag_id=my_parent_dag.dag_id + '.' + 'my_subdag',
        schedule_interval=my_parent_dag.schedule_interval,
        start_date=my_parent_dag.start_date),
    dag=my_parent_dag
)


task2 = BashOperator(
    task_id="bash_task",
    bash_command="echo hello world from parent",
    dag=my_parent_dag
)

task1 >> task2