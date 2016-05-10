from example import Tips, User, Employee
from peewee import fn, SelectQuery

selections = [
    Employee.first_name,
    Employee.department_id,
    Employee.salary,
    fn.AVG(Employee.salary).over(partition_by=[Employee.department_id]).alias('SalaryPerDepartament')
]
print selections

query = Tips.select(Tips.type, User.yelping_since, User.user_id, fn.Count('*').alias('num_tips'))\
    .join(User, on=(Tips.user_id == User.user_id))

print query

employ_in_1 = (Employee.department_id == 1)

query = SelectQuery(Employee, [employ_in_1])
