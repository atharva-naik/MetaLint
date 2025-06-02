#!/bin/python3
# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for,send_from_directory
from werkzeug.utils import secure_filename
import os
import re
import sh
from logsrv import logInfo,dingDing

basepath = os.path.dirname(__file__)

app = Flask(__name__)


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/jenkins_upload', methods=['GET'])
def index():
    url_path = request.full_path
    res = re.findall("tips",url_path)
    if res:
        return render_template('upload.html', tips="上传成功")

    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    md5 = request.form.get('md5sum')
    md5_small = md5.lower()

    dirs = "%s/code_files/%s" %(basepath,md5_small)
    if os.path.exists(dirs):
        sh.rm("-rf", dirs)


    f = request.files['file']
    sh.mkdir("%s/code_files/%s" %(basepath,md5_small))

    logInfo("[%s] [%s]  [%s]" %(md5,md5_small,f.filename))

    upload_path = os.path.join(basepath, 'code_files', md5_small,secure_filename(f.filename))  
    f.save(upload_path)
    
    dingDing(md5_small,f.filename)

    return redirect(url_for('index', tips="上传完成"))

if __name__ == '__main__':
    app.run(host="0.0.0.0",threaded=False,processes=5,debug=True)



