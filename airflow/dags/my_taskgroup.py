from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.dummy import DummyOperator
from airflow.utils.task_group import TaskGroup

# Création du DAG
with DAG(
    dag_id="task_group",
    schedule_interval=None,
    tags=['tutorial', 'datascientest'],
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(0, 1)
    }
) as my_dag:

    # Définition des tâches
    with TaskGroup("group_A_B") as group_A_B:
        with TaskGroup("group_A") as group_A:
            task_A1 = DummyOperator(task_id="task_A1")
            task_A2 = DummyOperator(task_id="task_A2")
            task_A3 = DummyOperator(task_id="task_A3")

        with TaskGroup("group_B") as group_B:
            task_B1 = DummyOperator(task_id="task_B1")
            task_B2 = DummyOperator(task_id="task_B2")

    with TaskGroup("group_C") as group_C:
        task_C1 = DummyOperator(task_id="task_C1")
        task_C2 = DummyOperator(task_id="task_C2")
        task_C3 = DummyOperator(task_id="task_C3")

    # Définition de l'ordre des tâches
    start_task = DummyOperator(task_id='start_task')
    end_task = DummyOperator(task_id='end_task')

    start_task >> group_A_B >> group_C >> end_task   