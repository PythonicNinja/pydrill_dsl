# -*- coding: utf-8 -*-
# CREATED ON DATE: 19.02.2016
from bisect import bisect_left, bisect_right
from copy import deepcopy

import re

from pydrill_dsl.query_objects import SelectQuery, Field, Node, QueryCompiler


# Python 2/3 compatibility helpers. These helpers are used internally and are
# not exported.
def with_metaclass(meta, base=object):
    return meta("NewBase", (base,), {})


class DoesNotExist(Exception): pass


class Drill(object):
    commit_select = False
    compiler_class = QueryCompiler
    quote_char = '`'
    quote_entity = '`'
    interpolation = '%s'
    field_overrides = {}
    op_overrides = {}

    def compiler(self):
        return self.compiler_class(
            self.quote_char, self.interpolation, self.field_overrides,
            self.op_overrides, self.quote_entity)


default_database = Drill


class _SortedFieldList(object):
    __slots__ = ('_keys', '_items')

    def __init__(self):
        self._keys = []
        self._items = []

    def __getitem__(self, i):
        return self._items[i]

    def __iter__(self):
        return iter(self._items)

    def __contains__(self, item):
        k = item._sort_key
        i = bisect_left(self._keys, k)
        j = bisect_right(self._keys, k)
        return item in self._items[i:j]

    def index(self, field):
        return self._keys.index(field._sort_key)

    def insert(self, item):
        k = item._sort_key
        i = bisect_left(self._keys, k)
        self._keys.insert(i, k)
        self._items.insert(i, item)

    def remove(self, item):
        idx = self.index(item)
        del self._items[idx]
        del self._keys[idx]


class ModelOptions(object):
    def __init__(self, cls, database=None, db_table=None, db_table_func=None,
                 indexes=None, order_by=None, primary_key=None,
                 table_alias=None, constraints=None, schema=None,
                 validate_backrefs=True, only_save_dirty=False, **kwargs):
        self.model_class = cls
        self.name = cls.__name__.lower()
        self.fields = {}
        self.columns = {}
        self.defaults = {}
        self._default_by_name = {}
        self._default_dict = {}
        self._default_callables = {}
        self._default_callable_list = []
        self._sorted_field_list = _SortedFieldList()
        self.sorted_fields = []
        self.sorted_field_names = []
        self.database = database or default_database()
        self.db_table = db_table
        self.db_table_func = db_table_func
        self.indexes = list(indexes or [])
        self.order_by = order_by
        self.primary_key = primary_key
        self.table_alias = table_alias
        self.constraints = constraints
        self.schema = schema
        self.validate_backrefs = validate_backrefs
        self.only_save_dirty = only_save_dirty

        self.auto_increment = None
        self.composite_key = False
        self.rel = {}
        self.reverse_rel = {}

        for key, value in kwargs.items():
            setattr(self, key, value)
        self._additional_keys = set(kwargs.keys())

        if self.db_table_func and not self.db_table:
            self.db_table = self.db_table_func(cls)

    def prepared(self):
        if self.order_by:
            norm_order_by = []
            for item in self.order_by:
                # if isinstance(item, Field):
                #     prefix = '-' if item._ordering == 'DESC' else ''
                #     item = prefix + item.name
                field = self.fields[item.lstrip('-')]
                if item.startswith('-'):
                    norm_order_by.append(field.desc())
                else:
                    norm_order_by.append(field.asc())
            self.order_by = norm_order_by

    def add_field(self, field):
        self.remove_field(field.name)
        self.fields[field.name] = field
        self.columns[field.db_column] = field

        self._sorted_field_list.insert(field)
        self.sorted_fields = list(self._sorted_field_list)
        self.sorted_field_names = [f.name for f in self.sorted_fields]

        if field.default is not None:
            self.defaults[field] = field.default
            if callable(field.default):
                self._default_callables[field] = field.default
                self._default_callable_list.append((field.name, field.default))
            else:
                self._default_dict[field] = field.default
                self._default_by_name[field.name] = field.default

    def remove_field(self, field_name):
        if field_name not in self.fields:
            return
        original = self.fields.pop(field_name)
        del self.columns[original.db_column]
        self._sorted_field_list.remove(original)
        self.sorted_fields = list(self._sorted_field_list)
        self.sorted_field_names = [f.name for f in self.sorted_fields]

        if original.default is not None:
            del self.defaults[original]
            if self._default_callables.pop(original, None):
                for i, (name, _) in enumerate(self._default_callable_list):
                    if name == field_name:
                        self._default_callable_list.pop(i)
                        break
            else:
                self._default_dict.pop(original, None)
                self._default_by_name.pop(original.name, None)

    def get_default_dict(self):
        dd = self._default_by_name.copy()
        for field_name, default in self._default_callable_list:
            dd[field_name] = default()
        return dd

    def get_field_index(self, field):
        try:
            return self._sorted_field_list.index(field)
        except ValueError:
            return -1

    def get_primary_key_fields(self):
        if self.composite_key:
            return [
                self.fields[field_name]
                for field_name in self.primary_key.field_names]
        return [self.primary_key]

    def rel_for_model(self, model, field_obj=None, multi=False):
        is_field = isinstance(field_obj, Field)
        is_node = not is_field and isinstance(field_obj, Node)
        if multi:
            accum = []
        # for field in self.sorted_fields:
        #     if isinstance(field, ForeignKeyField) and field.rel_model == model:
        #         is_match = (
        #             (field_obj is None) or
        #             (is_field and field_obj.name == field.name) or
        #             (is_node and field_obj._alias == field.name))
        #         if is_match:
        #             if not multi:
        #                 return field
        #             accum.append(field)
        if multi:
            return accum

    def reverse_rel_for_model(self, model, field_obj=None, multi=False):
        return model._meta.rel_for_model(self.model_class, field_obj, multi)

    def rel_exists(self, model):
        return self.rel_for_model(model) or self.reverse_rel_for_model(model)

    def related_models(self, backrefs=False):
        models = []
        stack = [self.model_class]
        while stack:
            model = stack.pop()
            if model in models:
                continue
            models.append(model)
            for fk in model._meta.rel.values():
                stack.append(fk.rel_model)
            if backrefs:
                for fk in model._meta.reverse_rel.values():
                    stack.append(fk.model_class)
        return models


class BaseModel(type):
    inheritable = set([
        'constraints', 'database', 'db_table_func', 'indexes', 'order_by',
        'primary_key', 'schema', 'validate_backrefs', 'only_save_dirty'])

    def __new__(cls, name, bases, attrs):
        if not bases:
            return super(BaseModel, cls).__new__(cls, name, bases, attrs)

        meta_options = {}
        meta = attrs.pop('Meta', None)
        if meta:
            for k, v in meta.__dict__.items():
                if not k.startswith('_'):
                    meta_options[k] = v

        model_pk = getattr(meta, 'primary_key', None)
        parent_pk = None

        # inherit any field descriptors by deep copying the underlying field
        # into the attrs of the new model, additionally see if the bases define
        # inheritable model options and swipe them
        for b in bases:
            if not hasattr(b, '_meta'):
                continue

            base_meta = getattr(b, '_meta')
            if parent_pk is None:
                parent_pk = deepcopy(base_meta.primary_key)
            all_inheritable = cls.inheritable | base_meta._additional_keys
            for (k, v) in base_meta.__dict__.items():
                if k in all_inheritable and k not in meta_options:
                    meta_options[k] = v

            for (k, v) in b.__dict__.items():
                if k in attrs:
                    continue
                    # if isinstance(v, FieldDescriptor):
                    #     if not v.field.primary_key:
                    #         attrs[k] = deepcopy(v.field)

        # initialize the new class and set the magic attributes
        cls = super(BaseModel, cls).__new__(cls, name, bases, attrs)
        ModelOptionsBase = meta_options.get('model_options_base', ModelOptions)
        cls._meta = ModelOptionsBase(cls, **meta_options)
        cls._data = None
        cls._meta.indexes = list(cls._meta.indexes)

        if not cls._meta.db_table:
            cls._meta.db_table = re.sub('[^\w]+', '_', cls.__name__.lower())

        # replace fields with field descriptors, calling the add_to_class hook
        fields = []
        for name, attr in cls.__dict__.items():
            if isinstance(attr, Field):
                if attr.primary_key and model_pk:
                    raise ValueError('primary key is overdetermined.')
                elif attr.primary_key:
                    model_pk, pk_name = attr, name
                else:
                    fields.append((attr, name))

        for field, name in fields:
            field.add_to_class(cls, name)

        # create a repr and error class before finalizing
        if hasattr(cls, '__unicode__'):
            setattr(cls, '__repr__', lambda self: '<%s: %r>' % (
                cls.__name__, self.__unicode__()))

        exc_name = '%sDoesNotExist' % cls.__name__
        exc_attrs = {'__module__': cls.__module__}
        exception_class = type(exc_name, (DoesNotExist,), exc_attrs)
        cls.DoesNotExist = exception_class
        cls._meta.prepared()

        if hasattr(cls, 'validate_model'):
            cls.validate_model()

        return cls

    def __iter__(self):
        return iter(self.select())


class Entity(Node):
    """A quoted-name or entity, e.g. "table"."column"."""
    _node_type = 'entity'

    def __init__(self, *path):
        super(Entity, self).__init__()
        self.path = path

    def clone_base(self):
        return Entity(*self.path)

    def __getattr__(self, attr):
        return Entity(*filter(None, self.path + (attr,)))


class Resource(with_metaclass(BaseModel)):
    def __init__(self, *args, **kwargs):
        self._data = self._meta.get_default_dict()
        self._dirty = set()
        self._obj_cache = {}

        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def select(cls, *selection):
        query = SelectQuery(cls, *selection)
        if cls._meta.order_by:
            query = query.order_by(*cls._meta.order_by)
        return query

    @classmethod
    def as_entity(cls):
        return Entity(cls._meta.storage_plugin, cls._meta.path)
