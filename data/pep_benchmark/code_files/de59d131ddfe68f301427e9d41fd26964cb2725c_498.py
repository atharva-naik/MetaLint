# -*- encoding: utf-8 -*-

import os

# Diretório base do projeto.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Configure com as informações do App do Twitter.
TWITTER_CONSUMER_KEY = ''
TWITTER_CONSUMER_SECRET = ''
TWITTER_ACCESS_TOKEN = ''
TWITTER_TOKEN_SECRET = ''
TWITTER_SEARCH_TERM = '"assistindo"'

# Nome do banco de dados no mongo.
MONGO_DB = os.getenv('MONGO_DB') or 'watching'

# URL de conexão com o mongo.
MONGO_URI = os.getenv('MONGO_URI') or 'mongodb://127.0.0.1:27017/%s' % MONGO_DB


try:
    # Sobrescreve as configurações pelo valore em "watching.instance.config".
    from watching.local_config import *
except ImportError:
    pass