import os

# Définition des variables d'environnement
os.environ['ACCESS_TOKEN_AUTH_0'] = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJBbGljZSJ9.NlaamftPNAgtReF0kY03XiDWplViB3DFfuqjnZ0Dy48'
os.environ['ACCESS_TOKEN_AUTH_1'] = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJKb2huIn0.HW2PSY6qVAPqtOi49Kf-bHh52e30BmvdQmqiC25KctY'
os.environ['ACCESS_TOKEN_AUTH_2'] = 'eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJGYWRpbWF0b3UifQ.r43zrSm_B3l5-xNjf7Q9XZXOQncGuI9YzarapOA0Wgg'
os.environ['MYSQL_ROOT_PWD'] = 'Rakuten'

# Vérifier si les variables ont été correctement définies et exportées
print(os.environ['ACCESS_TOKEN_AUTH_0'])

