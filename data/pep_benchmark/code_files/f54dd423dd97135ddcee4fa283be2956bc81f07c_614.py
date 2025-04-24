import sqlite3

from flask import Flask
from flask import g
from flask import render_template
from flask import request
from flask import Response

# アプリを定義
app = Flask(__name__)


# データベースとのセッションを定義
# （他の関数から呼び出せるように、g(global)にデータベースを代入している）
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('test_sqlite.db')
    return db


# データベースとのセッションが切れたら自動でdisconnectする関数を定義
@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()


# ルーティング
@app.route('/employee', methods=['POST', 'PUT', 'DELETE'])
@app.route('/employee/<name>', methods=['GET']) # <neme>がURLにあれば変数nameに代入
def employee(name=None):
    db = get_db()
    curs = db.cursor()
    curs.execute(
        'CREATE TABLE IF NOT EXISTS persons('
        'id INTEGER PRIMARY KEY AUTOINCREMENT, name STRING)'
    )
    db.commit()

    # URLから取得した名前をさらにnameと言う変数に格納する
    name = request.values.get('name', name)
    if request.method == 'GET':
        curs.execute('SELECT * FROM persons WHERE name = "{}"'.format(name))
        person = curs.fetchone()
        if not person:
            return "No", 404
        user_id, name = person
        return 'get {}: {}'.format(user_id, name), 200

    if request.method == 'POST':
        curs.execute('INSERT INTO persons(name) values("{}")'.format(name))
        db.commit()
        return 'created {}'.format(name), 201

    if request.method == 'PUT':
        new_name = request.values['new_name']
        curs.execute('UPDATE persons set name = "{}" WHERE name = "{}"'.format(
            new_name, name
        ))
        db.commit()
        return 'updated {}: {}'.format(name, new_name), 200

    if request.method == 'DELETE':
        curs.execute('DELETE FROM persons WHERE name ="{}"'.format(name))
        db.commit()
        return 'deleted {}'.format(name), 200

# ルーティング（ルートパス）
@app.route('/')
def hello_world():
    return 'top!'


@app.route('/hello/')
@app.route('/hello/<username>')
def hello_world2(username=None):
    # return
    return render_template('hello.html', username=username)


@app.route('/post', methods=['POST', 'PUT', 'DELETE'])
def show_post():
    return str(request.values['username'])


def main():
    # デバッグモードに設定
    app.debug = True
    # サーバー実行
    app.run()
    # app.run(host='1270.0.1', port=5000) # flaskのデフォルト設定

# お約束のやつ（このpythonファイルがimportされた時に勝手に実行されないように）
if __name__ == '__main__':
    main()
