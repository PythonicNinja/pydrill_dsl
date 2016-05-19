#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from peewee import fn


def test_returns_instance(employee_class):
    """
    :type employee_class: pydrill_dsl.resource.Resource
    """
    item = employee_class.select().get()
    assert item.first_name != ''


def test_returned_instace_has_methods(employee_class):
    """
    :type employee_class: pydrill_dsl.resource.Resource
    """
    item = employee_class.select().get()
    assert item.salary_with_name() != ''


def test_select_count(employee_class):
    """
    :type employee_class: pydrill_dsl.resource.Resource
    """
    count = employee_class.select().count()
    assert count == ('EXPR$0', '1155')


def test_alias_count(employee_class):
    """
    :type employee_class: pydrill_dsl.resource.Resource
    """
    count = employee_class.select(fn.Count('*').alias('count_employee')).scalar(as_tuple=True)
    assert count == [('count_employee', '1155')]


def test_predicate_order_by(employee_class):
    """
    :type employee_class: pydrill_dsl.resource.Resource
    """
    salary_gte_17K = (employee_class.salary >= 17000)
    salary_lte_25K = (employee_class.salary <= 25000)
    result = employee_class.select().where(salary_gte_17K & salary_lte_25K).order_by(+employee_class.salary)  # ASC
    results = list(result)
    lowest, highest = results[0], results[-1]
    assert isinstance(lowest, employee_class)
    assert lowest.salary < highest.salary


@pytest.mark.parametrize('filter,expected_count', [
    ({'first_name__eq': 'Sheri'}, 1),
    ({'first_name__ne': 'Sheri'}, 1154),
    ({'salary__gte': 17000}, 17),
])
def test_django_style_lookup(employee_class, filter, expected_count):
    """
    :type employee_class: pydrill_dsl.resource.Resource
    """
    result = employee_class.select().filter(**filter)
    assert len(result) == expected_count


def test_raw_query(employee_class):
    """
    :type employee_class: pydrill_dsl.resource.Resource
    """
    result = list(employee_class.raw('SELECT salary FROM `cp`.`employee.json` ORDER BY salary LIMIT 1'))
    assert isinstance(result[0], employee_class)
    assert result[0].salary == '20.0'
