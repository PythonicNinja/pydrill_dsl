from peewee import ExtQueryResultWrapper, Field

try:
    import pyodbc
    PYODBC_AVAILABLE = True
except ImportError:
    PYODBC_AVAILABLE = False


class DrillQueryResultWrapper(ExtQueryResultWrapper):
    def initialize(self, description):
        model = self.model
        conv = []
        i = 0
        for values in description:
            i += 1
            field_name = values[0]
            field_obj = model._meta.columns.get(field_name, Field())
            column = field_obj.name
            func = field_obj.python_value
            conv.append((i, column, func))
        self.conv = conv

    def process_row(self, row):
        instance = self.model()
        if PYODBC_AVAILABLE and isinstance(row, pyodbc.Row):
            row = zip(self.conv, row)
        row = dict(row)
        for i, column, func in self.conv:
            setattr(instance, column, func(row.get(column)))
        instance._prepare_instance()
        setattr(instance, '__data', row)
        return instance
