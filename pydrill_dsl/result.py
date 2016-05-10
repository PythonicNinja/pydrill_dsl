from peewee import ExtQueryResultWrapper


class DrillQueryResultWrapper(ExtQueryResultWrapper):
    def initialize(self, description):
        model = self.model
        conv = []
        i = 0
        for name, value in description:
            i += 1
            field_obj = model._meta.columns[name]
            column = field_obj.name
            func = field_obj.python_value
            conv.append((i, column, func))
        self.conv = conv

    def process_row(self, row):
        instance = self.model()
        row = dict(row)
        for i, column, func in self.conv:
            setattr(instance, column, func(row[column]))
        instance._prepare_instance()
        setattr(instance, '__data', row)
        return instance
