import flask
import boto3
import botocore
import json
import datetime
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql.functions import func
import sys
import os
import random

MAX_REQUESTS_TO_AWS = 20
MAX_PROMPT_LENGTH = 200

class Config(object):
    DEBUG = False
    TESTING = False
    CSRF_ENABLED = True
    SECRET_KEY = os.environ['DB_SECRET_KEY']
    SQLALCHEMY_DATABASE_URI = os.environ['DATABASE_URL']

app = flask.Flask(__name__, template_folder='templates')

aws_key = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_key = os.environ['AWS_SECRET_ACCESS_KEY']
app.config.from_object('app.Config')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
names = ['Саша', 'Игорек', 'Люда', 'Илья', 'Миша', 'Дима', 'Вова', 'Миклуха', 'Артем'];

from models import Journal
def get_count(q):
    count_q = q.statement.with_only_columns([func.count()]).order_by(None)
    count = q.session.execute(count_q).scalar()
    return count

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return flask.render_template('main.html')
    try:
        prompt = flask.request.form['message'][-MAX_PROMPT_LENGTH:]
        #cfg = botocore.config.Config(retries={'max_attempts': 0}, read_timeout=360, connect_timeout=360, region_name="eu-central-1" )
        if flask.request.form['requestid']:
            submitted = flask.request.form['submitted']
            requestid = flask.request.form['requestid']
            sqs = boto3.resource('sqs', region_name='eu-central-1', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key)
            queue = sqs.get_queue_by_name(QueueName = 'TabaquiQueue')
            for message in queue.receive_messages(MessageAttributeNames=['RequestId']):
                if message.message_attributes is not None:
                    mrequestid = message.message_attributes.get('RequestId').get('StringValue')
                    print(mrequestid + " " + requestid)
                    if mrequestid == requestid:
                        res = message.body
                        message.delete()
                        return flask.render_template('main.html', result = submitted + message.body)        
            return flask.render_template('main.html', result = prompt + '... ', submitted = submitted, requestid = requestid)
        db.session.query(Journal).filter(func.date(Journal.request) <= datetime.date.today() - datetime.timedelta(days=1)).delete(synchronize_session='fetch')
        db.session.commit()
        q = db.session.query(Journal.id)#.filter(...).order_by(...)
        if get_count(q) > MAX_REQUESTS_TO_AWS:
            return flask.render_template('main.html', result = 'Please try tomorrow')
        journal = Journal()
        db.session.add(journal)
        db.session.commit()
        db.session.refresh(journal)
        requestid = str(journal.id)
        print('Request id: ' + requestid)
        #boto3.setup_default_session(region_name='us-east-1')
        client = boto3.client('lambda', region_name='eu-central-1', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret_key) #config = cfg
        prompt = ''.join([(s.strip() if s.find(':') > 0 and s.find('"') > 0 else 
            names[random.randrange(0, len(names))] + ': "' + s.strip() + ' "') + '\n' 
            for s in prompt.split('\n') if s.strip()])
        if prompt.endswith(': "\n'):
            prompt = prompt[:-1]
        else:
            prompt = prompt + names[random.randrange(0, len(names))] + ': "'
        payload={"RequestId": requestid, "Prompt": prompt, "Temperature": 0.9}#, "NQuotes": 1}
        response = client.invoke(FunctionName = 'tabaqui_response', InvocationType = 'Event', LogType  = 'Tail', Payload = json.dumps(payload))
        #dictj = json.loads(response['Payload'].read().decode())
        return flask.render_template('main.html', result = prompt + "\n" + str(response['StatusCode']) + "\nHave to wait for request " + requestid, submitted = prompt, requestid = requestid)
    except:
        return flask.render_template('main.html', result = str(sys.exc_info()))#[0]

        
if __name__ == '__main__':
    app.run()#(host='0.0.0.0', port = os.environ['PORT'])