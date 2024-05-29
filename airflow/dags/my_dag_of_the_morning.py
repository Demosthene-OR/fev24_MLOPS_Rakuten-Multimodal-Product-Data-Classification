from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id='my_dag_of_the_morning',
    description='My DAG to know what to do in the morning',
    tags=['tutorial', 'datascientest'],
    schedule_interval=None,
    default_args={
        'owner': 'airflow',
        'start_date': days_ago(2),
    }
) as my_dag:

    def print_text(text):
        print(text)

    texts = [
        'Enfiler pantalon',
        'Enfiler chaussette droite',
        'Enfiler chaussure droite',
        'Enfiler chaussette gauche',
        'Enfiler chaussure gauche',
        'Sortir'
    ]

    ids = [
        'pantalon',
        'chaussette_droite',
        'chaussure_droite',
        'chaussette_gauche',
        'chaussure_gauche',
        'sortir'
    ]

    tasks = []
    for t, i in zip(texts, ids):
        task = PythonOperator(
            task_id=i,
            python_callable=print_text,
            op_kwargs={'text': t}
        )
        tasks.append(task)

tasks[0] >> [tasks[1], tasks[3]]
tasks[1] >> tasks[2]
tasks[3] >> tasks[4]
[tasks[2], tasks[4]] >> tasks[5]