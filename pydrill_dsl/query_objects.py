# -*- coding: utf-8 -*-
import operator
import sys

import re
from collections import deque, namedtuple

# from .resource import Resource
from inspect import isclass

try:
    from pydrill_dsl._speedups import strip_parens
except ImportError:
    def strip_parens(s):
        # Quick sanity check.
        if not s or s[0] != '(':
            return s

        ct = i = 0
        l = len(s)
        while i < l:
            if s[i] == '(' and s[l - 1] == ')':
                ct += 1
                i += 1
                l -= 1
            else:
                break
        if ct:
            # If we ever end up with negatively-balanced parentheses, then we
            # know that one of the outer parentheses was required.
            unbalanced_ct = 0
            required = 0
            for i in range(ct, l - ct):
                if s[i] == '(':
                    unbalanced_ct += 1
                elif s[i] == ')':
                    unbalanced_ct -= 1
                if unbalanced_ct < 0:
                    required += 1
                    unbalanced_ct = 0
                if required == ct:
                    break
            ct -= required
        if ct > 0:
            return s[ct:-ct]
        return s

PY2 = sys.version_info[0] == 2
PY3 = sys.version_info[0] == 3
PY26 = sys.version_info[:2] == (2, 6)
if PY3:
    import builtins
    from collections import Callable
    from functools import reduce

    callable = lambda c: isinstance(c, Callable)
    unicode_type = str
    string_type = bytes
    basestring = str
    print_ = getattr(builtins, 'print')
    binary_construct = lambda s: bytes(s.encode('raw_unicode_escape'))


    def reraise(tp, value, tb=None):
        if value.__traceback__ is not tb:
            raise value.with_traceback(tb)
        raise value
elif PY2:
    unicode_type = unicode
    string_type = basestring
    binary_construct = buffer


    def print_(s):
        sys.stdout.write(s)
        sys.stdout.write('\n')


    exec('def reraise(tp, value, tb=None): raise tp, value, tb')
else:
    raise RuntimeError('Unsupported python version.')


class _CDescriptor(object):
    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return Entity(instance._alias)
        return self


class attrdict(dict):
    def __getattr__(self, attr):
        return self[attr]


# Helper functions that are used in various parts of the codebase.
def merge_dict(source, overrides):
    merged = source.copy()
    merged.update(overrides)
    return merged


class AliasMap(object):
    prefix = 't'

    def __init__(self, start=0):
        self._alias_map = {}
        self._counter = start

    def __repr__(self):
        return '<AliasMap: %s>' % self._alias_map

    def add(self, obj, alias=None):
        if obj in self._alias_map:
            return
        self._counter += 1
        self._alias_map[obj] = alias or '%s%s' % (self.prefix, self._counter)

    def __getitem__(self, obj):
        if obj not in self._alias_map:
            self.add(obj)
        return self._alias_map[obj]

    def __contains__(self, obj):
        return obj in self._alias_map

    def update(self, alias_map):
        if alias_map:
            for obj, alias in alias_map._alias_map.items():
                if obj not in self:
                    self._alias_map[obj] = alias
        return self


# Operators used in binary expressions.
OP = attrdict(
    AND='and',
    OR='or',
    ADD='+',
    SUB='-',
    MUL='*',
    DIV='/',
    BIN_AND='&',
    BIN_OR='|',
    XOR='^',
    MOD='%',
    EQ='=',
    LT='<',
    LTE='<=',
    GT='>',
    GTE='>=',
    NE='!=',
    IN='in',
    NOT_IN='not in',
    IS='is',
    IS_NOT='is not',
    LIKE='like',
    ILIKE='ilike',
    BETWEEN='between',
    REGEXP='regexp',
    CONCAT='||',
)

JOIN = attrdict(
    INNER='INNER',
    LEFT_OUTER='LEFT OUTER',
    RIGHT_OUTER='RIGHT OUTER',
    FULL='FULL',
)
JOIN_INNER = JOIN.INNER
JOIN_LEFT_OUTER = JOIN.LEFT_OUTER
JOIN_FULL = JOIN.FULL

RESULTS_NAIVE = 1
RESULTS_MODELS = 2
RESULTS_TUPLES = 3
RESULTS_DICTS = 4
RESULTS_AGGREGATE_MODELS = 5

DJANGO_MAP = {
    'eq': OP.EQ,
    'lt': OP.LT,
    'lte': OP.LTE,
    'gt': OP.GT,
    'gte': OP.GTE,
    'ne': OP.NE,
    'in': OP.IN,
    'is': OP.IS,
    'like': OP.LIKE,
    'ilike': OP.ILIKE,
    'regexp': OP.REGEXP,
}


def returns_clone(func):
    """
    Method decorator that will "clone" the object before applying the given
    method.  This ensures that state is mutated in a more predictable fashion,
    and promotes the use of method-chaining.
    """

    def inner(self, *args, **kwargs):
        clone = self.clone()  # Assumes object implements `clone`.
        func(clone, *args, **kwargs)
        return clone

    inner.call_local = func  # Provide a way to call without cloning.
    return inner


# Classes representing the query tree.

class Node(object):
    """Base-class for any part of a query which shall be composable."""
    c = _CDescriptor()
    _node_type = 'node'

    def __init__(self):
        self._negated = False
        self._alias = None
        self._bind_to = None
        self._ordering = None  # ASC or DESC.

    @classmethod
    def extend(cls, name=None, clone=False):
        def decorator(method):
            method_name = name or method.__name__
            if clone:
                method = returns_clone(method)
            setattr(cls, method_name, method)
            return method

        return decorator

    def clone_base(self):
        return type(self)()

    def clone(self):
        inst = self.clone_base()
        inst._negated = self._negated
        inst._alias = self._alias
        inst._ordering = self._ordering
        inst._bind_to = self._bind_to
        return inst

    @returns_clone
    def __invert__(self):
        self._negated = not self._negated

    @returns_clone
    def alias(self, a=None):
        self._alias = a

    @returns_clone
    def bind_to(self, bt):
        """
        Bind the results of an expression to a specific model type. Useful
        when adding expressions to a select, where the result of the expression
        should be placed on a joined instance.
        """
        self._bind_to = bt

    @returns_clone
    def asc(self):
        self._ordering = 'ASC'

    @returns_clone
    def desc(self):
        self._ordering = 'DESC'

    def __pos__(self):
        return self.asc()

    def __neg__(self):
        return self.desc()

    def _e(op, inv=False):
        """
        Lightweight factory which returns a method that builds an Expression
        consisting of the left-hand and right-hand operands, using `op`.
        """

        def inner(self, rhs):
            if inv:
                return Expression(rhs, op, self)
            return Expression(self, op, rhs)

        return inner

    __and__ = _e(OP.AND)
    __or__ = _e(OP.OR)

    __add__ = _e(OP.ADD)
    __sub__ = _e(OP.SUB)
    __mul__ = _e(OP.MUL)
    __div__ = __truediv__ = _e(OP.DIV)
    __xor__ = _e(OP.XOR)
    __radd__ = _e(OP.ADD, inv=True)
    __rsub__ = _e(OP.SUB, inv=True)
    __rmul__ = _e(OP.MUL, inv=True)
    __rdiv__ = __rtruediv__ = _e(OP.DIV, inv=True)
    __rand__ = _e(OP.AND, inv=True)
    __ror__ = _e(OP.OR, inv=True)
    __rxor__ = _e(OP.XOR, inv=True)

    def __eq__(self, rhs):
        if rhs is None:
            return Expression(self, OP.IS, None)
        return Expression(self, OP.EQ, rhs)

    def __ne__(self, rhs):
        if rhs is None:
            return Expression(self, OP.IS_NOT, None)
        return Expression(self, OP.NE, rhs)

    __lt__ = _e(OP.LT)
    __le__ = _e(OP.LTE)
    __gt__ = _e(OP.GT)
    __ge__ = _e(OP.GTE)
    __lshift__ = _e(OP.IN)
    __rshift__ = _e(OP.IS)
    __mod__ = _e(OP.LIKE)
    __pow__ = _e(OP.ILIKE)

    bin_and = _e(OP.BIN_AND)
    bin_or = _e(OP.BIN_OR)

    # Special expressions.
    def in_(self, rhs):
        return Expression(self, OP.IN, rhs)

    def not_in(self, rhs):
        return Expression(self, OP.NOT_IN, rhs)

    def is_null(self, is_null=True):
        if is_null:
            return Expression(self, OP.IS, None)
        return Expression(self, OP.IS_NOT, None)

    def contains(self, rhs):
        return Expression(self, OP.ILIKE, '%%%s%%' % rhs)

    def startswith(self, rhs):
        return Expression(self, OP.ILIKE, '%s%%' % rhs)

    def endswith(self, rhs):
        return Expression(self, OP.ILIKE, '%%%s' % rhs)

    def between(self, low, high):
        return Expression(self, OP.BETWEEN, Clause(low, R('AND'), high))

    def regexp(self, expression):
        return Expression(self, OP.REGEXP, expression)

    def concat(self, rhs):
        return Expression(self, OP.CONCAT, rhs)


class SQL(Node):
    """An unescaped SQL string, with optional parameters."""
    _node_type = 'sql'

    def __init__(self, value, *params):
        self.value = value
        self.params = params
        super(SQL, self).__init__()

    def clone_base(self):
        return SQL(self.value, *self.params)


R = SQL  # backwards-compat.


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


class Func(Node):
    """An arbitrary SQL function call."""
    _node_type = 'func'

    def __init__(self, name, *arguments):
        self.name = name
        self.arguments = arguments
        self._coerce = True
        super(Func, self).__init__()

    @returns_clone
    def coerce(self, coerce=True):
        self._coerce = coerce

    def clone_base(self):
        res = Func(self.name, *self.arguments)
        res._coerce = self._coerce
        return res

    def over(self, partition_by=None, order_by=None, window=None):
        if isinstance(partition_by, Window) and window is None:
            window = partition_by
        if window is None:
            sql = Window(
                partition_by=partition_by, order_by=order_by).__sql__()
        else:
            sql = SQL(window._alias)
        return Clause(self, SQL('OVER'), sql)

    def __getattr__(self, attr):
        def dec(*args, **kwargs):
            return Func(attr, *args, **kwargs)

        return dec


# fn is a factory for creating `Func` objects and supports a more friendly
# API.  So instead of `Func("LOWER", param)`, `fn.LOWER(param)`.
fn = Func(None)


class Expression(Node):
    """A binary expression, e.g `foo + 1` or `bar < 7`."""
    _node_type = 'expression'

    def __init__(self, lhs, op, rhs, flat=False):
        super(Expression, self).__init__()
        self.lhs = lhs
        self.op = op
        self.rhs = rhs
        self.flat = flat

    def clone_base(self):
        return Expression(self.lhs, self.op, self.rhs, self.flat)


class Param(Node):
    """
    Arbitrary parameter passed into a query. Instructs the query compiler to
    specifically treat this value as a parameter, useful for `list` which is
    special-cased for `IN` lookups.
    """
    _node_type = 'param'

    def __init__(self, value, conv=None):
        self.value = value
        self.conv = conv
        super(Param, self).__init__()

    def clone_base(self):
        return Param(self.value, self.conv)


class Passthrough(Param):
    _node_type = 'passthrough'


class Clause(Node):
    """A SQL clause, one or more Node objects joined by spaces."""
    _node_type = 'clause'

    glue = ' '
    parens = False

    def __init__(self, *nodes, **kwargs):
        if 'glue' in kwargs:
            self.glue = kwargs['glue']
        if 'parens' in kwargs:
            self.parens = kwargs['parens']
        super(Clause, self).__init__()
        self.nodes = list(nodes)

    def clone_base(self):
        clone = Clause(*self.nodes)
        clone.glue = self.glue
        clone.parens = self.parens
        return clone


class CommaClause(Clause):
    """One or more Node objects joined by commas, no parens."""
    glue = ', '


class EnclosedClause(CommaClause):
    """One or more Node objects joined by commas and enclosed in parens."""
    parens = True


class Window(Node):
    def __init__(self, partition_by=None, order_by=None):
        super(Window, self).__init__()
        self.partition_by = partition_by
        self.order_by = order_by
        self._alias = self._alias or 'w'

    def __sql__(self):
        over_clauses = []
        if self.partition_by:
            over_clauses.append(Clause(
                SQL('PARTITION BY'),
                CommaClause(*self.partition_by)))
        if self.order_by:
            over_clauses.append(Clause(
                SQL('ORDER BY'),
                CommaClause(*self.order_by)))
        return EnclosedClause(Clause(*over_clauses))

    def clone_base(self):
        return Window(self.partition_by, self.order_by)


def Check(value):
    return SQL('CHECK (%s)' % value)


class DQ(Node):
    """A "django-style" filter expression, e.g. {'foo__eq': 'x'}."""

    def __init__(self, **query):
        super(DQ, self).__init__()
        self.query = query

    def clone_base(self):
        return DQ(**self.query)


class _StripParens(Node):
    _node_type = 'strip_parens'

    def __init__(self, node):
        super(_StripParens, self).__init__()
        self.node = node


class FieldDescriptor(object):
    # Fields are exposed as descriptors in order to control access to the
    # underlying "raw" data.
    def __init__(self, field):
        self.field = field
        self.att_name = self.field.name

    def __get__(self, instance, instance_type=None):
        if instance is not None:
            return instance._data.get(self.att_name)
        return self.field

    def __set__(self, instance, value):
        instance._data[self.att_name] = value
        instance._dirty.add(self.att_name)


class Field(Node):
    """A column on a table."""
    _field_counter = 0
    _order = 0
    _node_type = 'field'
    db_field = 'unknown'

    def __init__(self, null=False, index=False, unique=False,
                 verbose_name=None, help_text=None, db_column=None,
                 default=None, choices=None, primary_key=False, sequence=None,
                 constraints=None, schema=None):
        self.null = null
        self.index = index
        self.unique = unique
        self.verbose_name = verbose_name
        self.help_text = help_text
        self.db_column = db_column
        self.default = default
        self.choices = choices  # Used for metadata purposes, not enforced.
        self.primary_key = primary_key
        self.sequence = sequence  # Name of sequence, e.g. foo_id_seq.
        self.constraints = constraints  # List of column constraints.
        self.schema = schema  # Name of schema, e.g. 'public'.

        # Used internally for recovering the order in which Fields were defined
        # on the Model class.
        Field._field_counter += 1
        self._order = Field._field_counter
        self._sort_key = (self.primary_key and 1 or 2), self._order

        self._is_bound = False  # Whether the Field is "bound" to a Model.
        super(Field, self).__init__()

    def clone_base(self, **kwargs):
        inst = type(self)(
            null=self.null,
            index=self.index,
            unique=self.unique,
            verbose_name=self.verbose_name,
            help_text=self.help_text,
            db_column=self.db_column,
            default=self.default,
            choices=self.choices,
            primary_key=self.primary_key,
            sequence=self.sequence,
            constraints=self.constraints,
            schema=self.schema,
            **kwargs)
        if self._is_bound:
            inst.name = self.name
            inst.model_class = self.model_class
        inst._is_bound = self._is_bound
        return inst

    def add_to_class(self, model_class, name):
        """
        Hook that replaces the `Field` attribute on a class with a named
        `FieldDescriptor`. Called by the metaclass during construction of the
        `Model`.
        """
        self.name = name
        self.model_class = model_class
        self.db_column = self.db_column or self.name
        if not self.verbose_name:
            self.verbose_name = re.sub('_+', ' ', name).title()

        model_class._meta.add_field(self)
        setattr(model_class, name, FieldDescriptor(self))
        self._is_bound = True

    def get_database(self):
        return self.model_class._meta.database

    def get_column_type(self):
        field_type = self.get_db_field()
        return self.get_database().compiler().get_column_type(field_type)

    def get_db_field(self):
        return self.db_field

    def get_modifiers(self):
        return None

    def coerce(self, value):
        return value

    def db_value(self, value):
        """Convert the python value for storage in the database."""
        return value if value is None else self.coerce(value)

    def python_value(self, value):
        """Convert the database value to a pythonic value."""
        return value if value is None else self.coerce(value)

    def as_entity(self, with_table=False):
        if with_table:
            return Entity(self.model_class._meta.db_table, self.db_column)
        return Entity(self.db_column)

    def __ddl_column__(self, column_type):
        """Return the column type, e.g. VARCHAR(255) or REAL."""
        modifiers = self.get_modifiers()
        if modifiers:
            return SQL(
                '%s(%s)' % (column_type, ', '.join(map(str, modifiers))))
        return SQL(column_type)

    def __ddl__(self, column_type):
        """Return a list of Node instances that defines the column."""
        ddl = [self.as_entity(), self.__ddl_column__(column_type)]
        if not self.null:
            ddl.append(SQL('NOT NULL'))
        if self.primary_key:
            ddl.append(SQL('PRIMARY KEY'))
        if self.sequence:
            ddl.append(SQL("DEFAULT NEXTVAL('%s')" % self.sequence))
        if self.constraints:
            ddl.extend(self.constraints)
        return ddl

    def __hash__(self):
        return hash(self.name + '.' + self.model_class.__name__)


class ModelAlias(object):
    def __init__(self, model_class):
        self.__dict__['model_class'] = model_class

    def __getattr__(self, attr):
        model_attr = getattr(self.model_class, attr)
        # if isinstance(model_attr, Field):
        #     return FieldProxy(self, model_attr)
        return model_attr

    def __setattr__(self, attr, value):
        raise AttributeError('Cannot set attributes on ModelAlias instances')

    # def get_proxy_fields(self):
    #     return [
    #         FieldProxy(self, f) for f in self.model_class._meta.sorted_fields]

    def select(self, *selection):
        if not selection:
            selection = self.get_proxy_fields()
        query = SelectQuery(self, *selection)
        if self._meta.order_by:
            query = query.order_by(*self._meta.order_by)
        return query

    def __call__(self, **kwargs):
        return self.model_class(**kwargs)


JoinMetadata = namedtuple('JoinMetadata', (
    'src_model',  # Source Model class.
    'dest_model',  # Dest Model class.
    'src',  # Source, may be Model, ModelAlias
    'dest',  # Dest, may be Model, ModelAlias, or SelectQuery.
    'attr',  # Attribute name joined instance(s) should be assigned to.
    'primary_key',  # Primary key being joined on.
    'foreign_key',  # Foreign key being joined from.
    'is_backref',  # Is this a backref, i.e. 1 -> N.
    'alias',  # Explicit alias given to join expression.
    'is_self_join',  # Is this a self-join?
    'is_expression',  # Is the join ON clause an Expression?
))


class Join(namedtuple('_Join', ('src', 'dest', 'join_type', 'on'))):
    def get_foreign_key(self, source, dest, field=None):
        if isinstance(source, SelectQuery) or isinstance(dest, SelectQuery):
            return None, None
        fk_field = source._meta.rel_for_model(dest, field)
        if fk_field is not None:
            return fk_field, False
        reverse_rel = source._meta.reverse_rel_for_model(dest, field)
        if reverse_rel is not None:
            return reverse_rel, True
        return None, None

    def get_join_type(self):
        return self.join_type or JOIN.INNER

    def model_from_alias(self, model_or_alias):
        if isinstance(model_or_alias, ModelAlias):
            return model_or_alias.model_class
        elif isinstance(model_or_alias, SelectQuery):
            return model_or_alias.model_class
        return model_or_alias

    def _join_metadata(self):
        # Get the actual tables being joined.
        src = self.model_from_alias(self.src)
        dest = self.model_from_alias(self.dest)

        join_alias = isinstance(self.on, Node) and self.on._alias or None
        is_expression = isinstance(self.on, (Expression, Func, SQL))

        on_field = None  # isinstance(self.on, (Field, FieldProxy)) and self.on or None
        if on_field:
            fk_field = on_field
            is_backref = on_field.name not in src._meta.fields
        else:
            fk_field, is_backref = self.get_foreign_key(src, dest, self.on)
            if fk_field is None and self.on is not None:
                fk_field, is_backref = self.get_foreign_key(src, dest)

        if fk_field is not None:
            primary_key = fk_field.to_field
        else:
            primary_key = None

        if not join_alias:
            if fk_field is not None:
                if is_backref:
                    target_attr = dest._meta.db_table
                else:
                    target_attr = fk_field.name
            else:
                try:
                    target_attr = self.on.lhs.name
                except AttributeError:
                    target_attr = dest._meta.db_table
        else:
            target_attr = None

        return JoinMetadata(
            src_model=src,
            dest_model=dest,
            src=self.src,
            dest=self.dest,
            attr=join_alias or target_attr,
            primary_key=primary_key,
            foreign_key=fk_field,
            is_backref=is_backref,
            alias=join_alias,
            is_self_join=src is dest,
            is_expression=is_expression)

    @property
    def metadata(self):
        if not hasattr(self, '_cached_metadata'):
            self._cached_metadata = self._join_metadata()
        return self._cached_metadata


class QueryCompiler(object):
    # Mapping of `db_type` to actual column type used by database driver.
    # Database classes may provide additional column types or overrides.
    field_map = {
        'bare': '',
        'bigint': 'BIGINT',
        'blob': 'BLOB',
        'bool': 'SMALLINT',
        'date': 'DATE',
        'datetime': 'DATETIME',
        'decimal': 'DECIMAL',
        'double': 'REAL',
        'fixed_char': 'CHAR',
        'float': 'REAL',
        'int': 'INTEGER',
        'primary_key': 'INTEGER',
        'smallint': 'SMALLINT',
        'string': 'VARCHAR',
        'text': 'TEXT',
        'time': 'TIME',
    }

    # Mapping of OP. to actual SQL operation.  For most databases this will be
    # the same, but some column types or databases may support additional ops.
    # Like `field_map`, Database classes may extend or override these.
    op_map = {
        OP.EQ: '=',
        OP.LT: '<',
        OP.LTE: '<=',
        OP.GT: '>',
        OP.GTE: '>=',
        OP.NE: '<>',
        OP.IN: 'IN',
        OP.NOT_IN: 'NOT IN',
        OP.IS: 'IS',
        OP.IS_NOT: 'IS NOT',
        OP.BIN_AND: '&',
        OP.BIN_OR: '|',
        OP.LIKE: 'LIKE',
        OP.ILIKE: 'ILIKE',
        OP.BETWEEN: 'BETWEEN',
        OP.ADD: '+',
        OP.SUB: '-',
        OP.MUL: '*',
        OP.DIV: '/',
        OP.XOR: '#',
        OP.AND: 'AND',
        OP.OR: 'OR',
        OP.MOD: '%',
        OP.REGEXP: 'REGEXP',
        OP.CONCAT: '||',
    }

    join_map = {
        JOIN.INNER: 'INNER JOIN',
        JOIN.LEFT_OUTER: 'LEFT OUTER JOIN',
        JOIN.RIGHT_OUTER: 'RIGHT OUTER JOIN',
        JOIN.FULL: 'FULL JOIN',
    }
    alias_map_class = AliasMap

    def __init__(self, quote_char='"', interpolation='?', field_overrides=None,
                 op_overrides=None, quote_entity=None):
        self.quote_char = quote_char
        self.quote_entity = quote_entity or quote_char
        self.interpolation = interpolation
        self._field_map = merge_dict(self.field_map, field_overrides or {})
        self._op_map = merge_dict(self.op_map, op_overrides or {})
        self._parse_map = self.get_parse_map()
        self._unknown_types = set(['param'])

    def get_parse_map(self):
        # To avoid O(n) lookups when parsing nodes, use a lookup table for
        # common node types O(1).
        return {
            'expression': self._parse_expression,
            'param': self._parse_param,
            'passthrough': self._parse_param,
            'func': self._parse_func,
            'clause': self._parse_clause,
            'entity': self._parse_entity,
            'field': self._parse_field,
            'sql': self._parse_sql,
            'select_query': self._parse_select_query,
            'compound_select_query': self._parse_compound_select_query,
            'strip_parens': self._parse_strip_parens,
        }

    def quote(self, s, quote_char=None):
        if not quote_char:
            quote_char = self.quote_char
        return '%s%s%s' % (quote_char, s, quote_char)

    def get_column_type(self, f):
        return self._field_map[f] if f in self._field_map else f.upper()

    def get_op(self, q):
        return self._op_map[q]

    def _sorted_fields(self, field_dict):
        return sorted(field_dict.items(), key=lambda i: i[0]._sort_key)

    def _parse_default(self, node, alias_map, conv):
        return self.interpolation, [node]

    def _parse_expression(self, node, alias_map, conv):
        if isinstance(node.lhs, Field):
            conv = node.lhs
        lhs, lparams = self.parse_node(node.lhs, alias_map, conv)
        rhs, rparams = self.parse_node(node.rhs, alias_map, conv)
        template = '%s %s %s' if node.flat else '(%s %s %s)'
        sql = template % (lhs, self.get_op(node.op), rhs)
        return sql, lparams + rparams

    def _parse_param(self, node, alias_map, conv):
        if node.conv:
            params = [node.conv(node.value)]
        else:
            params = [node.value]
        return self.interpolation, params

    def _parse_func(self, node, alias_map, conv):
        conv = node._coerce and conv or None
        sql, params = self.parse_node_list(node.arguments, alias_map, conv)
        return '%s(%s)' % (node.name, strip_parens(sql)), params

    def _parse_clause(self, node, alias_map, conv):
        sql, params = self.parse_node_list(
            node.nodes, alias_map, conv, node.glue)
        if node.parens:
            sql = '(%s)' % strip_parens(sql)
        return sql, params

    def _parse_entity(self, node, alias_map, conv):
        return '.'.join(map(self.quote, node.path)), []

    def _parse_sql(self, node, alias_map, conv):
        return node.value, list(node.params)

    def _parse_field(self, node, alias_map, conv):
        if alias_map:
            sql = '.'.join(map(self.quote, (
                alias_map[node.model_class],
                node.db_column)))
        else:
            sql = self.quote(node.db_column)
        return sql, []

    def _parse_compound_select_query(self, node, alias_map, conv):
        csq = 'compound_select_query'
        if node.rhs._node_type == csq and node.lhs._node_type != csq:
            first_q, second_q = node.rhs, node.lhs
            inv = True
        else:
            first_q, second_q = node.lhs, node.rhs
            inv = False

        new_map = self.alias_map_class()
        if first_q._node_type == csq:
            new_map._counter = alias_map._counter

        first, first_p = self.generate_select(first_q, new_map)
        second, second_p = self.generate_select(
            second_q,
            self.calculate_alias_map(second_q, new_map))

        if inv:
            l, lp, r, rp = second, second_p, first, first_p
        else:
            l, lp, r, rp = first, first_p, second, second_p

        # We add outer parentheses in the event the compound query is used in
        # the `from_()` clause, in which case we'll need them.
        if node.database.compound_select_parentheses:
            sql = '((%s) %s (%s))' % (l, node.operator, r)
        else:
            sql = '(%s %s %s)' % (l, node.operator, r)
        return sql, lp + rp

    def _parse_select_query(self, node, alias_map, conv):
        clone = node.clone()
        if not node._explicit_selection:
            select_field = clone.model_class._meta.primary_key
            clone._select = (select_field,)
        sub, params = self.generate_select(clone, alias_map)
        return '(%s)' % strip_parens(sub), params

    def _parse_strip_parens(self, node, alias_map, conv):
        sql, params = self.parse_node(node.node, alias_map, conv)
        return strip_parens(sql), params

    def _parse(self, node, alias_map, conv):
        # By default treat the incoming node as a raw value that should be
        # parameterized.
        node_type = getattr(node, '_node_type', None)
        unknown = False

        from pydrill_dsl.resource import Resource

        if node_type in self._parse_map:
            sql, params = self._parse_map[node_type](node, alias_map, conv)
            unknown = node_type in self._unknown_types
        elif isinstance(node, (list, tuple)):
            # If you're wondering how to pass a list into your query, simply
            # wrap it in Param().
            sql, params = self.parse_node_list(node, alias_map, conv)
            sql = '(%s)' % sql
        elif (isclass(node) and issubclass(node, Resource)) or \
            isinstance(node, ModelAlias):
            entity = node.as_entity().alias(alias_map[node])
            sql, params = self.parse_node(entity, alias_map, conv)
        else:
            sql, params = self._parse_default(node, alias_map, conv)
            unknown = True
        return sql, params, unknown

    def parse_node(self, node, alias_map=None, conv=None):
        sql, params, unknown = self._parse(node, alias_map, conv)
        if unknown and conv and params:
            params = [conv.db_value(i) for i in params]

        if isinstance(node, Node):
            if node._negated:
                sql = 'NOT %s' % sql
            if node._alias:
                sql = ' '.join((sql, 'AS', node._alias))
            if node._ordering:
                sql = ' '.join((sql, node._ordering))
        return sql, params

    def parse_node_list(self, nodes, alias_map, conv=None, glue=', '):
        sql = []
        params = []
        for node in nodes:
            node_sql, node_params = self.parse_node(node, alias_map, conv)
            sql.append(node_sql)
            params.extend(node_params)
        return glue.join(sql), params

    def calculate_alias_map(self, query, alias_map=None):
        new_map = self.alias_map_class()
        if alias_map is not None:
            new_map._counter = alias_map._counter

        new_map.add(query.model_class, query.model_class._meta.table_alias)
        for src_model, joined_models in query._joins.items():
            new_map.add(src_model, src_model._meta.table_alias)
            for join_obj in joined_models:
                if isinstance(join_obj.dest, Node):
                    new_map.add(join_obj.dest, join_obj.dest.alias)
                else:
                    new_map.add(join_obj.dest, join_obj.dest._meta.table_alias)

        return new_map.update(alias_map)

    def build_query(self, clauses, alias_map=None):
        return self.parse_node(Clause(*clauses), alias_map)

    def generate_joins(self, joins, model_class, alias_map):
        # Joins are implemented as an adjancency-list graph. Perform a
        # depth-first search of the graph to generate all the necessary JOINs.

        clauses = []
        seen = set()
        q = [model_class]
        while q:
            curr = q.pop()
            if curr not in joins or curr in seen:
                continue
            seen.add(curr)
            for join in joins[curr]:
                src = curr
                dest = join.dest
                if isinstance(join.on, (Expression, Func, Clause, Entity)):
                    # Clear any alias on the join expression.
                    constraint = join.on.clone().alias()
                else:
                    metadata = join.metadata
                    if metadata.is_backref:
                        fk_model = join.dest
                        pk_model = join.src
                    else:
                        fk_model = join.src
                        pk_model = join.dest

                    fk = metadata.foreign_key
                    if fk:
                        lhs = getattr(fk_model, fk.name)
                        rhs = getattr(pk_model, fk.to_field.name)
                        if metadata.is_backref:
                            lhs, rhs = rhs, lhs
                        constraint = (lhs == rhs)
                    else:
                        raise ValueError('Missing required join predicate.')

                if isinstance(dest, Node):
                    # TODO: ensure alias?
                    dest_n = dest
                else:
                    q.append(dest)
                    dest_n = dest.as_entity().alias(alias_map[dest])

                join_type = join.get_join_type()
                if join_type in self.join_map:
                    join_sql = SQL(self.join_map[join_type])
                else:
                    join_sql = SQL(join_type)
                clauses.append(
                    Clause(join_sql, dest_n, SQL('ON'), constraint))

        return clauses

    def generate_select(self, query, alias_map=None):
        model = query.model_class
        db = model._meta.database

        alias_map = self.calculate_alias_map(query, alias_map)

        if False:  # and isinstance(query, CompoundSelect):
            clauses = [_StripParens(query)]
        else:
            if not query._distinct:
                clauses = [SQL('SELECT')]
            else:
                clauses = [SQL('SELECT DISTINCT')]
                if query._distinct not in (True, False):
                    clauses += [SQL('ON'), EnclosedClause(*query._distinct)]

            select_clause = Clause(*query._select)
            select_clause.glue = ', '

            clauses.extend((select_clause, SQL('FROM')))
            if query._from is None:
                clauses.append(model.as_entity().alias(alias_map[model]))
            else:
                clauses.append(CommaClause(*query._from))

        if query._windows is not None:
            clauses.append(SQL('WINDOW'))
            clauses.append(CommaClause(*[
                Clause(
                    SQL(window._alias),
                    SQL('AS'),
                    window.__sql__())
                for window in query._windows]))

        join_clauses = self.generate_joins(query._joins, model, alias_map)
        if join_clauses:
            clauses.extend(join_clauses)

        if query._where is not None:
            clauses.extend([SQL('WHERE'), query._where])

        if query._group_by:
            clauses.extend([SQL('GROUP BY'), CommaClause(*query._group_by)])

        if query._having:
            clauses.extend([SQL('HAVING'), query._having])

        if query._order_by:
            clauses.extend([SQL('ORDER BY'), CommaClause(*query._order_by)])

        if query._limit or (query._offset and db.limit_max):
            limit = query._limit or db.limit_max
            clauses.append(SQL('LIMIT %s' % limit))
        if query._offset:
            clauses.append(SQL('OFFSET %s' % query._offset))

        for_update, no_wait = query._for_update
        if for_update:
            stmt = 'FOR UPDATE NOWAIT' if no_wait else 'FOR UPDATE'
            clauses.append(SQL(stmt))

        return self.build_query(clauses, alias_map)

    def _get_field_clause(self, fields, clause_type=EnclosedClause):
        return clause_type(*[
            field.as_entity(with_table=False) for field in fields])

    def field_definition(self, field):
        column_type = self.get_column_type(field.get_db_field())
        ddl = field.__ddl__(column_type)
        return Clause(*ddl)


class Query(Node):
    """Base class representing a database query on one or more tables."""
    require_commit = True

    def __init__(self, model_class):
        super(Query, self).__init__()

        self.model_class = model_class
        self.database = model_class._meta.database

        self._dirty = True
        self._query_ctx = model_class
        self._joins = {self.model_class: []}  # Join graph as adjacency list.
        self._where = None

    def __repr__(self):
        return '%s %s' % (self.model_class, self.sql())

    def clone(self):
        query = type(self)(self.model_class)
        query.database = self.database
        return self._clone_attributes(query)

    def _clone_attributes(self, query):
        if self._where is not None:
            query._where = self._where.clone()
        query._joins = self._clone_joins()
        query._query_ctx = self._query_ctx
        return query

    def _clone_joins(self):
        return dict(
            (mc, list(j)) for mc, j in self._joins.items())

    def _add_query_clauses(self, initial, expressions, conjunction=None):
        reduced = reduce(operator.and_, expressions)
        if initial is None:
            return reduced
        conjunction = conjunction or operator.and_
        return conjunction(initial, reduced)

    def _model_shorthand(self, args):
        accum = []
        for arg in args:
            if isinstance(arg, Node):
                accum.append(arg)
            elif isinstance(arg, Query):
                accum.append(arg)
            elif isinstance(arg, ModelAlias):
                accum.extend(arg.get_proxy_fields())
                # elif isclass(arg) and issubclass(arg, Model):
                #     accum.extend(arg._meta.sorted_fields)
        return accum

    @returns_clone
    def where(self, *expressions):
        self._where = self._add_query_clauses(self._where, expressions)

    @returns_clone
    def orwhere(self, *expressions):
        self._where = self._add_query_clauses(
            self._where, expressions, operator.or_)

    @returns_clone
    def join(self, dest, join_type=None, on=None):
        src = self._query_ctx
        if not on:
            require_join_condition = (
                isinstance(dest, SelectQuery) or
                (isclass(dest) and not src._meta.rel_exists(dest)))
            if require_join_condition:
                raise ValueError('A join condition must be specified.')
        elif isinstance(on, basestring):
            on = src._meta.fields[on]
        self._joins.setdefault(src, [])
        self._joins[src].append(Join(src, dest, join_type, on))
        if not isinstance(dest, SelectQuery):
            self._query_ctx = dest

    @returns_clone
    def switch(self, model_class=None):
        """Change or reset the query context."""
        self._query_ctx = model_class or self.model_class

    def ensure_join(self, lm, rm, on=None):
        ctx = self._query_ctx
        for join in self._joins.get(lm, []):
            if join.dest == rm:
                return self
        return self.switch(lm).join(rm, on=on).switch(ctx)

    def convert_dict_to_node(self, qdict):
        accum = []
        joins = []
        relationship = ()  # (ForeignKeyField, ReverseRelationDescriptor)
        for key, value in sorted(qdict.items()):
            curr = self.model_class
            if '__' in key and key.rsplit('__', 1)[1] in DJANGO_MAP:
                key, op = key.rsplit('__', 1)
                op = DJANGO_MAP[op]
            else:
                op = OP.EQ
            for piece in key.split('__'):
                model_attr = getattr(curr, piece)
                if isinstance(model_attr, relationship):
                    curr = model_attr.rel_model
                    joins.append(model_attr)
            accum.append(Expression(model_attr, op, value))
        return accum, joins

    def filter(self, *args, **kwargs):
        # normalize args and kwargs into a new expression
        dq_node = Node()
        if args:
            dq_node &= reduce(operator.and_, [a.clone() for a in args])
        if kwargs:
            dq_node &= DQ(**kwargs)

        # dq_node should now be an Expression, lhs = Node(), rhs = ...
        q = deque([dq_node])
        dq_joins = set()
        while q:
            curr = q.popleft()
            if not isinstance(curr, Expression):
                continue
            for side, piece in (('lhs', curr.lhs), ('rhs', curr.rhs)):
                if isinstance(piece, DQ):
                    query, joins = self.convert_dict_to_node(piece.query)
                    dq_joins.update(joins)
                    expression = reduce(operator.and_, query)
                    # Apply values from the DQ object.
                    expression._negated = piece._negated
                    expression._alias = piece._alias
                    setattr(curr, side, expression)
                else:
                    q.append(piece)

        dq_node = dq_node.rhs

        query = self.clone()
        # for field in dq_joins:
        #     if isinstance(field, ForeignKeyField):
        #         lm, rm = field.model_class, field.rel_model
        #         field_obj = field
        #     elif isinstance(field, ReverseRelationDescriptor):
        #         lm, rm = field.field.rel_model, field.rel_model
        #         field_obj = field.field
        #     query = query.ensure_join(lm, rm, field_obj)
        return query.where(dq_node)

    def compiler(self):
        return self.database.compiler()

    def sql(self):
        raise NotImplementedError

    def _execute(self):
        sql = self.sql()
        return self.database.execute_sql(sql, self.require_commit)

    def execute(self):
        raise NotImplementedError

    def scalar(self, as_tuple=False, convert=False):
        if convert:
            row = self.tuples().first()
        else:
            row = self._execute().fetchone()
        if row and not as_tuple:
            return row[0]
        else:
            return row


class RawQuery(Query):
    """
    Execute a SQL query, returning a standard iterable interface that returns
    model instances.
    """

    def __init__(self, model, query, *params):
        self._sql = query
        self._params = list(params)
        self._qr = None
        self._tuples = False
        self._dicts = False
        super(RawQuery, self).__init__(model)

    def clone(self):
        query = RawQuery(self.model_class, self._sql, *self._params)
        query._tuples = self._tuples
        query._dicts = self._dicts
        return query

    # join = not_allowed('joining')
    # where = not_allowed('where')
    # switch = not_allowed('switch')

    @returns_clone
    def tuples(self, tuples=True):
        self._tuples = tuples

    @returns_clone
    def dicts(self, dicts=True):
        self._dicts = dicts

    def sql(self):
        return self._sql, self._params

    def execute(self):
        if self._qr is None:
            if self._tuples:
                QRW = self.database.get_result_wrapper(RESULTS_TUPLES)
            elif self._dicts:
                QRW = self.database.get_result_wrapper(RESULTS_DICTS)
            else:
                QRW = self.database.get_result_wrapper(RESULTS_NAIVE)
            self._qr = QRW(self.model_class, self._execute(), None)
        return self._qr

    def __iter__(self):
        return iter(self.execute())


class SelectQuery(Query):
    _node_type = 'select_query'

    def __init__(self, model_class, *selection):
        super(SelectQuery, self).__init__(model_class)
        # self.require_commit = self.database.commit_select
        self.__select(*selection)
        self._from = None
        self._group_by = None
        self._having = None
        self._order_by = None
        self._windows = None
        self._limit = None
        self._offset = None
        self._distinct = False
        self._for_update = (False, False)
        self._naive = False
        self._tuples = False
        self._dicts = False
        self._aggregate_rows = False
        self._alias = None
        self._qr = None

    def _clone_attributes(self, query):
        query = super(SelectQuery, self)._clone_attributes(query)
        query._explicit_selection = self._explicit_selection
        query._select = list(self._select)
        if self._from is not None:
            query._from = []
            for f in self._from:
                if isinstance(f, Node):
                    query._from.append(f.clone())
                else:
                    query._from.append(f)
        if self._group_by is not None:
            query._group_by = list(self._group_by)
        if self._having:
            query._having = self._having.clone()
        if self._order_by is not None:
            query._order_by = list(self._order_by)
        if self._windows is not None:
            query._windows = list(self._windows)
        query._limit = self._limit
        query._offset = self._offset
        query._distinct = self._distinct
        query._for_update = self._for_update
        query._naive = self._naive
        query._tuples = self._tuples
        query._dicts = self._dicts
        query._aggregate_rows = self._aggregate_rows
        query._alias = self._alias
        return query

    def compound_op(operator):
        def inner(self, other):
            supported_ops = self.model_class._meta.database.compound_operations
            if operator not in supported_ops:
                raise ValueError(
                    'Your database does not support %s' % operator)
                # return CompoundSelect(self.model_class, self, operator, other)

        return inner

    _compound_op_static = staticmethod(compound_op)
    __or__ = compound_op('UNION')
    __and__ = compound_op('INTERSECT')
    __sub__ = compound_op('EXCEPT')

    def __xor__(self, rhs):
        # Symmetric difference, should just be (self | rhs) - (self & rhs)...
        wrapped_rhs = self.model_class.select(SQL('*')).from_(
            EnclosedClause((self & rhs)).alias('_')).order_by()
        return (self | rhs) - wrapped_rhs

    def union_all(self, rhs):
        return SelectQuery._compound_op_static('UNION ALL')(self, rhs)

    def __select(self, *selection):
        print selection
        self._explicit_selection = len(selection) > 0
        selection = selection or self.model_class._meta.sorted_fields
        self._select = self._model_shorthand(selection)

    select = returns_clone(__select)

    @returns_clone
    def from_(self, *args):
        self._from = None
        if args:
            self._from = list(args)

    @returns_clone
    def group_by(self, *args):
        self._group_by = self._model_shorthand(args)

    @returns_clone
    def having(self, *expressions):
        self._having = self._add_query_clauses(self._having, expressions)

    @returns_clone
    def order_by(self, *args):
        self._order_by = list(args)

    @returns_clone
    def window(self, *windows):
        self._windows = list(windows)

    @returns_clone
    def limit(self, lim):
        self._limit = lim

    @returns_clone
    def offset(self, off):
        self._offset = off

    @returns_clone
    def paginate(self, page, paginate_by=20):
        if page > 0:
            page -= 1
        self._limit = paginate_by
        self._offset = page * paginate_by

    @returns_clone
    def distinct(self, is_distinct=True):
        self._distinct = is_distinct

    @returns_clone
    def for_update(self, for_update=True, nowait=False):
        self._for_update = (for_update, nowait)

    @returns_clone
    def naive(self, naive=True):
        self._naive = naive

    @returns_clone
    def tuples(self, tuples=True):
        self._tuples = tuples

    @returns_clone
    def dicts(self, dicts=True):
        self._dicts = dicts

    @returns_clone
    def aggregate_rows(self, aggregate_rows=True):
        self._aggregate_rows = aggregate_rows

    @returns_clone
    def alias(self, alias=None):
        self._alias = alias

    def annotate(self, rel_model, annotation=None):
        if annotation is None:
            annotation = fn.Count(rel_model._meta.primary_key).alias('count')
        if self._query_ctx == rel_model:
            query = self.switch(self.model_class)
        else:
            query = self.clone()
        query = query.ensure_join(query._query_ctx, rel_model)
        if not query._group_by:
            query._group_by = [x.alias() for x in query._select]
        query._select = tuple(query._select) + (annotation,)
        return query

    def _aggregate(self, aggregation=None):
        if aggregation is None:
            aggregation = fn.Count(SQL('*'))
        query = self.order_by()
        query._select = [aggregation]
        return query

    def aggregate(self, aggregation=None, convert=True):
        return self._aggregate(aggregation).scalar(convert=convert)

    def count(self, clear_limit=False):
        return self.wrapped_count(clear_limit=clear_limit)

    def wrapped_count(self, clear_limit=False):
        clone = self.order_by()
        if clear_limit:
            clone._limit = clone._offset = None

        sql = clone.sql()
        wrapped = 'SELECT COUNT(1) FROM (%s) AS wrapped_select' % sql
        self.sql = lambda: wrapped
        return self

    def exists(self):
        clone = self.paginate(1, 1)
        clone._select = [SQL('1')]
        return bool(clone.scalar())

    def get(self):
        clone = self.paginate(1, 1)
        try:
            return next(clone.execute())
        except StopIteration:
            raise self.model_class.DoesNotExist(
                'Instance matching query does not exist:\nSQL: %s\nPARAMS: %s'
                % self.sql())

    def first(self):
        res = self.execute()
        res.fill_cache(1)
        try:
            return res._result_cache[0]
        except IndexError:
            pass

    def sql(self):
        sql, params = self.compiler().generate_select(self)
        return sql % tuple(map(lambda s: "'" + str(s) + "'", params))

    def verify_naive(self):
        model_class = self.model_class
        for node in self._select:
            if isinstance(node, Field) and node.model_class != model_class:
                return False
            elif isinstance(node, Node) and node._bind_to is not None:
                if node._bind_to != model_class:
                    return False
        return True

    def get_query_meta(self):
        return (self._select, self._joins)

    def _get_result_wrapper(self):
        if self._tuples:
            return self.database.get_result_wrapper(RESULTS_TUPLES)
        elif self._dicts:
            return self.database.get_result_wrapper(RESULTS_DICTS)
        elif self._naive or not self._joins or self.verify_naive():
            return self.database.get_result_wrapper(RESULTS_NAIVE)
        elif self._aggregate_rows:
            return self.database.get_result_wrapper(RESULTS_AGGREGATE_MODELS)
        else:
            return self.database.get_result_wrapper(RESULTS_MODELS)

    def execute(self):
        if self._dirty or self._qr is None:
            model_class = self.model_class
            query_meta = self.get_query_meta()
            ResultWrapper = self._get_result_wrapper()
            self._qr = ResultWrapper(model_class, self._execute(), query_meta)
            self._dirty = False
            return self._qr
        else:
            return self._qr

    def __iter__(self):
        return iter(self.execute())

    def iterator(self):
        return iter(self.execute().iterator())

    def __getitem__(self, value):
        res = self.execute()
        if isinstance(value, slice):
            index = value.stop
        else:
            index = value
        if index is not None and index >= 0:
            index += 1
        res.fill_cache(index)
        return res._result_cache[value]

    def __len__(self):
        return len(self.execute())

    if PY3:
        def __hash__(self):
            return id(self)
