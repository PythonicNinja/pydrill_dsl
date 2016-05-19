from peewee import OP, Database, ImproperlyConfigured, QueryCompiler, logger

from pydrill_dsl.result import DrillQueryResultWrapper

try:
    from pydrill.client import PyDrill
    PYDRILL_AVAILABLE = True
except ImportError:
    PYDRILL_AVAILABLE = False

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False


class Drill(Database):
    commit_select = False
    compiler_class = QueryCompiler
    quote_char = '`'
    quote_entity = '`'
    interpolation = '%s'
    field_overrides = {}
    op_overrides = {
        OP.NE: '<>',
        OP.ILIKE: 'LIKE',
    }

    def get_result_wrapper(self, wrapper_type):
        return DrillQueryResultWrapper

    def compiler(self):
        return self.compiler_class(
            self.quote_char, self.interpolation, self.field_overrides,
            self.op_overrides)

    @classmethod
    def sql_fill_params(cls, sql, params):
        if params:
            sql = sql % tuple(map(lambda s: "'" + str(s) + "'", params))
        return sql

    def execute_sql(self, sql, params=None, require_commit=True):
        logger.debug((sql, params))
        with self.exception_wrapper():
            cursor = self.get_cursor()
            try:
                cursor.execute(Drill.sql_fill_params(sql, params))
            except Exception:
                if self.get_autocommit() and self.autorollback:
                    self.rollback()
                raise
            else:
                if require_commit and self.get_autocommit():
                    self.commit()
        return cursor

    def get_pydrill_conn(self, _connection_conf):
        if not PYDRILL_AVAILABLE:
            raise ImproperlyConfigured('pydrill must be installed.')
        conn = PyDrill(**_connection_conf)

        class cursor:
            def __init__(self):
                self.data = []
                self.description = ''
                self.cursor_index = -1

            def execute(self, sql):
                self.data = conn.query(sql)
                return self.data

            def fetchone(self, *args, **kwargs):
                self.cursor_index += 1
                try:
                    self.description = list(self.data.rows[self.cursor_index].items())
                    return self.description
                except IndexError:
                    return

            def close(self):
                return

        conn.cursor = cursor
        conn.commit = lambda: None
        return conn

    def get_pyodbc_conn(self, _connection_conf):
        if not PYODBC_AVAILABLE:
            raise ImproperlyConfigured('pyodbc must be installed.')
        conn = pyodbc.connect(_connection_conf['dsn'], autocommit=True)
        return conn

    def _connect(self, connection_conf, **kwargs):
        _connection_conf = dict(connection_conf, **kwargs)
        if 'dsn' in _connection_conf:
            return self.get_pyodbc_conn(_connection_conf)
        elif 'host' in _connection_conf and 'port' in _connection_conf:
            return self.get_pydrill_conn(_connection_conf)
