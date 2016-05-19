===============================
pyDrill-dsl
===============================

.. image:: https://img.shields.io/travis/PythonicNinja/pydrill_dsl.svg
        :target: https://travis-ci.org/PythonicNinja/pydrill_dsl

.. image:: https://img.shields.io/pypi/v/pydrill_dsl.svg
        :target: https://pypi.python.org/pypi/pydrill_dsl

.. image:: https://readthedocs.org/projects/pydrill_dsl/badge/?version=latest
        :target: https://readthedocs.org/projects/pydrill_dsl/?badge=latest
        :alt: Documentation Status

.. image:: https://coveralls.io/repos/github/PythonicNinja/pydrill_dsl/badge.svg
        :target: https://coveralls.io/github/PythonicNinja/pydrill_dsl


Pythonic DSL for `Apache Drill <https://drill.apache.org/>`_.

*Schema-free SQL Query Engine for Hadoop, NoSQL and Cloud Storage*

* Free software: MIT license
* Documentation: https://pydrill_dsl.readthedocs.org.

Features
--------

* Uses Peewee syntax
* Support for all storage plugins
* Support for drivers PyODBC and pyDrill

Sample usage
------------
::

    from pydrill_dsl.resource import Resource

    class Employee(Resource):
        first_name = Field()
        salary = Field()
        position_id = Field()
        department_id = Field()

        class Meta:
            storage_plugin = 'cp'
            path = 'employee.json'


    Employee.select().filter(salary__gte=17000)

    Employee.select().paginate(page=1, paginate_by=5)


    salary_gte_17K = (Employee.salary >= 17000)
    salary_lte_25K = (Employee.salary <= 25000)
    Employee.select().where(salary_gte_17K & salary_lte_25K)

    Employee.select(
        fn.Min(Employee.salary).alias('salary_min'),
        fn.Max(Employee.salary).alias('salary_max')
    ).scalar(as_tuple=True)

    # creation of resource can be done without creation of class:
    employee = Resource(storage_plugin='cp', path='employee.json',
                        fields=('first_name', 'salary', 'position_id', 'department_id'))
