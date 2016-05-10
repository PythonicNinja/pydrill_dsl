#!/usr/bin/env python
# -*- coding: utf-8 -*-
import unittest

from peewee import Field
from pydrill_dsl.resource import Resource


class TestPydrill_dsl(unittest.TestCase):

    def setUp(self):
        class Employee(Resource):
            first_name = Field()
            salary = Field()
            position_id = Field()
            department_id = Field()

            class Meta:
                storage_plugin = 'cp'
                path = 'employee.json'

            def salary_with_name(self):
                return u"{0} - {1}".format(self.first_name, self.salary)
        self.query_class = Employee

    def tearDown(self):
        pass

    def test_returns_instance(self):
        item = self.query_class.select().get()
        assert item.first_name != ''

    def test_returned_instace_has_methods(self):
        item = self.query_class.select().get()
        assert item.salary_with_name() != ''

if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
