import os

from peewee import with_metaclass, BaseModel, SelectQuery, Entity, RawQuery, Field
from pydrill_dsl.database import Drill


class Resource(with_metaclass(BaseModel)):
    def __init__(self, *args, **kwargs):
        self._data = self._meta.get_default_dict()
        self._dirty = set()
        self._obj_cache = {}

        for k, v in kwargs.items():
            setattr(self, k, v)

        class_name = kwargs.get('name')
        if class_name:
            cls = type(class_name, (Resource,), {})
            self.__class__ = cls

        for field_name in kwargs.get('fields', []):
            field = Field()
            field.add_to_class(self.__class__, field_name)
            setattr(self.__class__, field_name, field)

        kwargs_set_on_meta = ['storage_plugin', 'path']
        for kwarg_name in kwargs_set_on_meta:
            value = kwargs.get(kwarg_name)
            if value:
                setattr(self.__class__._meta, kwarg_name, value)

    @classmethod
    def select(cls, *selection):
        cls._meta.remove_field('id')
        query = SelectQuery(cls, *selection)
        if cls._meta.order_by:
            query = query.order_by(*cls._meta.order_by)
        return query

    @classmethod
    def raw(cls, sql, *params):
        return RawQuery(cls, sql, *params)

    @classmethod
    def as_entity(cls):
        return Entity(cls._meta.storage_plugin, cls._meta.path)

    def _prepare_instance(self, *args, **kwargs):
        pass

    class Meta:
        database = Drill({
            'host': os.environ.get('PYDRILL_HOST', 'localhost'),
            'port': os.environ.get('PYDRILL_PORT', 8047)
        })
