# -*- coding: utf8 -*-
import MySQLdb
from common.db import connect_db
from common.config import SESSION_KEY
from flask import Flask, g, session, request, abort, render_template, redirect, url_for, jsonify

app = Flask(__name__,
        template_folder = 'layouts',
        static_folder = 'static')
app.secret_key = SESSION_KEY

@app.before_request
def init():
    g.conn = connect_db()
    g.cursor = g.conn.cursor()

@app.teardown_request
def teardown(e):
    if hasattr(g, 'conn'):
        g.conn.close()

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.route('/')
def index():
    if 'name' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods = ['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    if request.method == 'POST':
        session['name'] = request.form['name']
        return redirect(url_for('index'))

@app.route('/pages')
def pages():
    if 'name' not in session:
        return redirect(url_for('login'))
    sql = "SELECT name, title, content FROM pages ORDER BY id DESC"
    g.cursor.execute(sql)
    return render_template('pages.html',
            pageList = [dict(zip(['name', 'title', 'content'], row))
                for row in g.cursor.fetchall()])

@app.route('/pages/<name>')
def page(name):
    if 'name' not in session:
        return redirect(url_for('login'))
    sql = "SELECT name, title, content, with_form FROM pages WHERE name = '%s'" % name
    if not g.cursor.execute(sql):
        abort(404)
    return render_template('page.html',
            **dict(zip(['name', 'title', 'content', 'with_form'],
                g.cursor.fetchone())))

@app.route('/form', methods = ['GET', 'POST'])
def form():
    if request.method == 'GET':
        return render_template('form.html', )
    else:
        for field in request.form:
            if not request.form[field]:
                return jsonify(err_code = -1, err_field = field,
                        msg = u'有未填的区域')
        if request.form['meizhi'] == 'false':
            return jsonify(err_code = -1, error_field = 'meizhi',
                    msg = u'有未填的区域')
        sql = """INSERT INTO form_data
        (name, email, commen, introduction, jiecao, meizhi)
        VALUES ('%s', '%s', '%s', '%s', '%s', '%s')""" % (request.form['name'],
                request.form['email'], request.form['comment'], request.form['introduction'],
                request.form['jiecao'], request.form['meizhi'])
        g.cursor.execute(sql)
        g.conn.commit()
        return jsonify(err_code = 0, msg = u'这不是真的笔试。。请尝试SQL注入。。')

@app.route('/admin', methods = ['GET', 'POST'])
def admin():
    if 'name' not in session:
        return redirect(url_for('login'))
    if request.method == 'GET':
        return render_template('admin_login.html')
    if request.method == 'POST':
        sql = "SELECT password from admin where username=%s"
        if g.cursor.execute(sql, request.form['username']):
            if g.cursor.fetchone()[0] == request.form['password']:
                sql = "INSERT INTO injectors (name, way) VALUES (%s, 1)"
                g.cursor.execute(sql, session['name'])
                g.conn.commit()
                session['injected'] = True
                return redirect(url_for('admin_index'))

        sql = "SELECT 1 FROM admin WHERE username='%s' AND password='%s'" % (
                request.form['username'], request.form['password'])
        if g.cursor.execute(sql):
            sql = "INSERT INTO injectors (name, way) VALUES (%s, 0)"
            g.cursor.execute(sql, session['name'])
            g.conn.commit()
            session['injected'] = True
            return redirect(url_for('admin_index'))

        return render_template('admin_login.html', err = True)

@app.route('/admin/index')
def admin_index():
    if 'injected' in session:
        sql = "SELECT * from injectors"
        g.cursor.execute(sql)
        return render_template('admin.html', injectors = [row for row in g.cursor.fetchall()])
    else:
        return redirect(url_for('admin'))