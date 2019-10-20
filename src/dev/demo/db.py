from sqlalchemy import create_engine
from sqlalchemy.sql import text
import json

# global access-keys
# todo. encrypted access-key (JWT?)
keys = json.load(open('demo/static/AccessKey.json'))

class DatabaseHelper:

    # default values
    username = 'aiu01',
    password = 'aiuser01!',
    hostname = '192.168.100.103',
    port = '1521',
    database = 'HIS031',
    charset = 'utf8',
    encoding = 'utf8',
    unicode_error = 'ignore'

    def __init__(self):
        self._username = self.username
        self._password = self.password
        self._hostname = self.hostname
        self._port = self.port
        self._database = self.database
        self._charset = self.charset
        self._encoding = self.encoding
        self._unicode_error = self.unicode_error

        # init engine
        oracle_connection_string = 'oracle://{username}:{password}@{hostname}:{port}/{database}'

        self._engine = create_engine(
        oracle_connection_string.format(
            username=self._username,
            password=self._password,
            hostname=self._hostname,
            port=self._port,
            database=self._database,
            charset=self._charset,
            encoding=self._encoding,
            unicode_error=self._unicode_error
        )
        )

class HIS_Database(DatabaseHelper):
    # connection information for HIS DB
    username = keys['HIS']['username']
    password = keys['HIS']['password']
    hostname = keys['HIS']['hostname']
    port = keys['HIS']['port']
    database = keys['HIS']['database']
    charset = keys['HIS']['charset']
    encoding = keys['HIS']['encoding']
    unicode_error = keys['HIS']['unicode_error']

    def update_status(self, **status):
        with self._engine.connect() as conn:
            statement = text("""
                    UPDATE AIMMREQP
                    SET STATUS='{STATUS}'
                        ,ERRMSG='{ERRMSG}'
                        ,LASTUPDTRID='{LASTUPDTRID}'
                        ,LASTUPDTDT={LASTUPDTDT}
                    WHERE GUBUN='{GUBUN}'
                    AND PID='{PID}'
                    AND SEQNO='{SEQNO}'
                """.format(**status))
            conn.execute(statement)

    def update_result(self, data):
        # write result to DB!
        with self._engine.connect() as con:
            statement = text("""INSERT INTO AIMMRSLT(GUBUN, PID, SEQNO, FILEKEY, FILESEQ, RSLTDT, RSLTCD, RSLTVAL, FSTRGSTRID, FSTRGSTDT) VALUES(:GUBUN, :PID, :SEQNO, :FILEKEY, :FILESEQ, :RSLTDT, :RSLTCD, :RSLTVAL, :FSTRGSTRID, systimestamp)""")
            con.execute(statement, *data)