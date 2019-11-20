import os
import datetime
from flask import jsonify


def bad_request(status, message):
    # print(message)
    res = jsonify({'status': status, 'message': message})
    res.status_code = 400
    return res


class VariableInterface:
    pass


class ExceptionLogger:
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):

        try:
            return self.func(*args, **kwargs)
        except Exception as e:
            VariableInterface.db.update_status(STATUS="ERR",
                                               ERRMSG=str(e),
                                               LASTUPDTRID=VariableInterface.USERID,
                                               LASTUPDTDT="SYSTIMESTAMP",
                                               GUBUN=VariableInterface.GUBUN,
                                               PID=VariableInterface.PID,
                                               SEQNO=VariableInterface.SEQNO,
                                               sess=VariableInterface.sess)

            return bad_request(status='ERR', message=str(e))


class BadRequestException(Exception):
    def __str__(self):
        return 'BadRequestError'


def check_response(logger):
    def wrapper(*args, **kwargs):
        response = logger.__call__(*args, **kwargs)
        if response is not None:
            if response.status_code == 400:
                VariableInterface.BadResponse = response
                raise BadRequestException
            elif response.status_code == 200:
                VariableInterface.GreetingResponse = response

    return wrapper


def FetchData_FromURL(url, path):
    status = os.system('wget {} -O {}'.format(url, path))
    if status != 0:
        raise Exception('Invalid URL, remote file does not exists.')


@check_response
@ExceptionLogger
def fetch_file(*args, **kwargs):
    if VariableInterface.query_res.empty:
        raise Exception('File is not found!')

    DATA_SUFFIX = VariableInterface.query_res.filepath.values[kwargs['fileseq']].replace(
        '\\', '/')
    URL = os.path.join(VariableInterface.DATA_PREFIX, DATA_SUFFIX)
    FetchData_FromURL(URL, VariableInterface.VIDEO_PATH)


@check_response
@ExceptionLogger
def run_model(*args, **kwargs):
    # update status
    VariableInterface.res = VariableInterface.runner.run(
        VariableInterface.VIDEO_PATH)


@check_response
@ExceptionLogger
def write_result(*args, **kwargs):
    # write result to DB!
    data = []
    curTime = str(datetime.datetime.now())
    for pname, pval in zip(*list(VariableInterface.res.values())):
        obj = {"GUBUN": VariableInterface.GUBUN,
               "PID": VariableInterface.PID,
               "SEQNO": VariableInterface.SEQNO,
               "FILEKEY": VariableInterface.FILEKEY,
               "FILESEQ": VariableInterface.FILESEQ,
               "RSLTDT": curTime,
               "RSLTCD": VariableInterface.codeNameTable[pname],
               "RSLTVAL": f'{pval:.4f}',
               "FSTRGSTRID": VariableInterface.USERID,
               "FSTRGSTDT": str(datetime.datetime.now())}
        data.append(obj)

    VariableInterface.db.update_result(data, sess=VariableInterface.sess)


@check_response
@ExceptionLogger
def success_callback(*args, **kwargs):
    # update status ('E') to notify task is done!
    VariableInterface.db.update_status(STATUS="E",
                                       ERRMSG="",
                                       LASTUPDTRID=VariableInterface.USERID,
                                       LASTUPDTDT="SYSTIMESTAMP",
                                       GUBUN=VariableInterface.GUBUN,
                                       PID=VariableInterface.PID,
                                       SEQNO=VariableInterface.SEQNO,
                                       sess=VariableInterface.sess)

    # success msg
    res = jsonify({'status': 'E', 'message': 'success'})
    res.status_code = 200

    return res
