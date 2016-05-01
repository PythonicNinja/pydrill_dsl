from pydrill_dsl.query_objects import QueryCompiler, SelectQuery
from example import Employee


employ_in_1 = Employee.department_id == 1

print employ_in_1

query = SelectQuery(Employee, [employ_in_1])

print

