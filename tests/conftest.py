import pytest

from peewee import Field
from pydrill.client import PyDrill
from pydrill_dsl.resource import Resource


@pytest.fixture(scope='function', autouse=True)
def pydrill_instance():
    drill = PyDrill()
    return drill


@pytest.fixture()
def pydrill_url(pydrill_instance):
    """
    :type pydrill_instance: pydrill.client.PyDrill
    """
    return pydrill_instance.transport.connection.base_url


@pytest.fixture(scope='function', autouse=True)
def employee_class():
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

    return Employee
