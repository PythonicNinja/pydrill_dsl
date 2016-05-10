from peewee import OP, Database, ImproperlyConfigured, QueryCompiler
from pydrill_dsl.result import DrillQueryResultWrapper

try:
    from pydrill.client import PyDrill

    PYDRILL_AVAILABLE = True
except ImportError:
    PYDRILL_AVAILABLE = False


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

    def _connect(self, connection_conf, **kwargs):
        if not PYDRILL_AVAILABLE:
            raise ImproperlyConfigured('pydrill must be installed.')
        _connection_conf = dict(connection_conf, **kwargs)
        conn = PyDrill(**_connection_conf)

        class cursor:
            def __init__(self):
                self.data = []
                self.description = ''
                self.cursor_index = -1

            def execute(self, sql, params):
                if params:
                    sql = sql % tuple(map(lambda s: "'" + str(s) + "'", params))
                self.data = conn.query(sql)
                return self.data

            def fetchone(self, *args, **kwargs):
                self.cursor_index += 1
                try:
                    self.description = self.data.rows[self.cursor_index].items()
                    return self.description
                except IndexError:
                    return

            def close(self):
                return

        conn.cursor = cursor
        conn.commit = lambda: None
        return conn
