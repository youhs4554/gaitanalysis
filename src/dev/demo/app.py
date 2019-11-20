# flask....

import os
import time

from flask import (Flask, flash, request, redirect,
                   render_template, session)
import json
from utils.generate_model import load_trained_ckpt
import pandas as pd
from demo.utils import (VariableInterface, fetch_file, run_model,
                        write_result, success_callback,
                        BadRequestException, bad_request)
from demo.db import HIS_Database


class Runner():
    def __init__(self, opt, net, localizer, interval_selector, worker,
                 spatial_transform, target_transform, target_columns):
        self.opt = opt
        self.localizer = localizer
        self.interval_selector = interval_selector
        self.worker = worker
        self.spatial_transform = spatial_transform
        self.target_transform = target_transform
        self.target_columns = target_columns

        self.net = load_trained_ckpt(opt, net)

    def run(self, path, startTime, endTime):
        t0 = time.time()
        y_pred = self.worker._run_demo(
            self.net, path, self.localizer,
            self.interval_selector,
            startTime, endTime,
            spatial_transform=self.spatial_transform['test'],
            target_transform=self.target_transform)
        print('runtime : ', time.time() - t0)

        return {'pos': self.target_columns,
                'val': y_pred[0].tolist()}


# model
runner = None

APP_HOME = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(APP_HOME, 'static')
ALLOWED_EXTENSIONS = set(['mp4', 'avi'])

app = Flask(__name__)
app.config['APP_HOME'] = APP_HOME
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'

# db instance
VariableInterface.db = HIS_Database()


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def main():
    return render_template('index.html', data={'filename': ""})


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))

    session['vname'] = file.filename

    return render_template('index.html',
                           data={'filename': os.path.join("static", file.filename)})


@app.route('/api')
def api_call():
    # from request.get
    VariableInterface.GUBUN = request.args['GUBUN']
    VariableInterface.PID = request.args['PID']
    VariableInterface.SEQNO = request.args['SEQNO']
    VariableInterface.FILEKEY = request.args['FILEKEY']
    VariableInterface.FILESEQ = VariableInterface.SEQNO
    VariableInterface.USERID = 'AIM'

    VariableInterface.DATA_PREFIX = "http://192.168.100.121/his031edu/attach"

    data = pd.read_sql("SELECT * FROM com.zfmmfile",
                       VariableInterface.db._engine)
    VariableInterface.query_res = data.query(
        "filekey==@VariableInterface.FILEKEY")

    VariableInterface.codeNameTable = json.load(
        open('./demo/static/codeNameTable.json'))

    # default response
    VariableInterface.response = bad_request(
        status='ERR', message='Unknown Error.')

    from sqlalchemy.orm import scoped_session, sessionmaker

    engine = VariableInterface.db._engine
    session = scoped_session(sessionmaker(
        autocommit=False, autoflush=False, bind=engine))

    sess = session()
    VariableInterface.sess = sess
    try:
        for n in range(len(VariableInterface.query_res.filepath.values)):
            VariableInterface.VIDEO_PATH = f'./demo/static/tmp_{n}.avi'

            # fetch video file!
            fetch_file(fileseq=n)

            # run model!
            run_model()

            # write result to DB!
            write_result()

        # callback of success, notifying task is done!
        success_callback()

        sess.commit()

    except BadRequestException:
        sess.rollback()
        return VariableInterface.BadResponse
    finally:
        sess.close()

    return VariableInterface.GreetingResponse


@app.route('/run')
def run():
    vname = session.get('vname', None)
    start = request.args.get('start', False, type=bool)
    startTime = eval(request.args.get('startTime'))
    endTime = eval(request.args.get('endTime'))

    if vname and start and not session.get('res', None):
        path = os.path.join(app.config['UPLOAD_FOLDER'], vname)

        # run model here?
        res = runner.run(path, startTime, endTime)
        session['res'] = res

        return '1'
    else:
        return '0'


@app.route('/stat')
def stat():
    res = session.pop('res', None)
    session.pop('vname', None)

    print('######', res)
    return render_template('status.html', data=json.dumps(res))


def set_runner(*args, **kwargs):
    global runner
    VariableInterface.runner = runner = Runner(*args, **kwargs)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=40000)
