# -*- coding: utf-8 -*-
from peewee import Field, fn
from pydrill_dsl.database import Drill

from pydrill_dsl.resource import Resource

YELP_LOCATION = '/Users/macbookair/Downloads/datasets/yelp_dataset_challenge_academic_dataset/'


class Employee(Resource):
    first_name = Field()
    salary = Field()
    position_id = Field()
    department_id = Field()

    class Meta:
        storage_plugin = 'cp'
        path = 'employee.json'
        database = Drill({
            'dsn': 'Driver=/opt/mapr/drillodbc/lib/universal/libmaprdrillodbc.dylib;ConnectionType=Direct;Host=127.0.0.1;Port=31010;Catalog=DRILL;AuthenticationType=Basic Authentication;AdvancedProperties=CastAnyToVarchar=true;HandshakeTimeout=5;QueryTimeout=180;TimestampTZDisplayTimezone=utc;ExcludedSchemas=sys,INFORMATION_SCHEMA;NumberOfPrefetchBuffers=5;UID=[USERNAME];PWD=[PASSWORD]'
        })

    def salary_with_name(self):
        return u"{} - {}".format(self.first_name, self.salary)


employee = Resource(storage_plugin='cp', path='employee.json',
                    fields=('first_name', 'salary', 'position_id', 'department_id'))

employe_ = Resource(name='EmployeeClass', storage_plugin='cp', path='employee.json',
                    fields=('first_name', 'salary', 'position_id', 'department_id'))

list(employee.select().where(employee.salary >= 2000))
x = employe_.select().where(employe_.salary >= 2000)
x.sql
list(x)


class YoublProducts(Resource):
    count = Field()
    shop = Field()
    date = Field()

    class Meta:
        storage_plugin = 'mongo.youbl'
        path = 'product_shop_count'


class Reviews(Resource):
    votes = Field()
    user_id = Field()
    review_id = Field()
    stars = Field()
    date = Field()
    text = Field()
    type = Field()
    business_id = Field()

    class Meta:
        storage_plugin = 'dfs'
        path = YELP_LOCATION + 'yelp_academic_dataset_user.json'


class User(Resource):
    yelping_since = Field()
    votes = Field()
    review_count = Field()
    name = Field()
    user_id = Field()
    friends = Field()
    fans = Field()
    average_stars = Field()
    type = Field()
    compliments = Field()
    elite = Field()

    class Meta:
        storage_plugin = 'dfs'
        path = YELP_LOCATION + 'yelp_academic_dataset_review.json'


class Tips(Resource):
    user_id = Field()
    # text = Field()
    type = Field()

    class Meta:
        storage_plugin = 'dfs'
        path = YELP_LOCATION + 'yelp_academic_dataset_tip.json'


if __name__ == '__main__':
    salary_gte_17K = (Employee.salary >= 17000)
    salary_lte_25K = (Employee.salary <= 25000)
    salary_gte_0K = (Employee.salary >= 0)
    num_tips = fn.COUNT(Tips.user_id)
    queries = [
                  Employee.select().filter(first_name__eq='Sheri'),
                  Employee.select().filter(first_name__ne='Sheri'),
                  Employee.select().filter(salary__gte=17000),
                  Employee.select(),
                  Employee.select(Employee.position_id).order_by(Employee.department_id),
                  Employee.select().paginate(page=1, paginate_by=5),
                  Employee.select().paginate(page=2, paginate_by=5),
                  Employee.select().paginate(page=3, paginate_by=5),
                  Employee.select().where((Employee.first_name == 'Sheri')),
                  Employee.select().where((Employee.first_name.contains('S'))),
                  Employee.select().where(salary_gte_17K & salary_lte_25K),
                  Employee.select().where(salary_gte_17K),
                  Employee.select().order_by((Employee.first_name)).limit(10).where(
                      (Employee.position_id == Employee.department_id) & salary_gte_17K),
                  Employee.select().count(),
                  Employee.select().where(fn.Lower(fn.Substr(Employee.first_name, 1, 1)) == 's').count(),
                  Employee.select().order_by(fn.Rand()).limit(3),
                  Employee.select(
                      fn.Min(Employee.salary).alias('salary_min'),
                      fn.Max(Employee.salary).alias('salary_max')
                  ).scalar(as_tuple=True),

              ] + [
                  # YoublProducts.select().distinct(
                  #     YoublProducts.shop).order_by(
                  #     YoublProducts.count.desc()
                  # ).limit(10),
                  # YoublProducts.select().paginate(page=1, paginate_by=100),
                  # YoublProducts.select().paginate(page=2, paginate_by=100),
                  # YoublProducts.select().paginate(page=3, paginate_by=100),
                  # YoublProducts.select().paginate(page=4, paginate_by=100),
              ] + [
                  # Tips.select().count(),
                  # Tips.select().where(Tips.type == "tip").count(),
                  # Tips.select(Tips.user_id, num_tips.alias('num_tips')).group_by(Tips.user_id).order_by(
                  #     num_tips.desc()).limit(10),
              ] + [
                  # Reviews.select(Reviews.votes).count(),
                  # Reviews.select(Reviews.votes).limit(10),
                  # Reviews.select().order_by(Reviews.votes.desc()).limit(10),
              ] + [
                  # Tips.select(Tips.type, User.yelping_since, User.user_id, fn.Count('*').alias('num_tips'))
                  #     .join(User, on=(Tips.user_id == User.user_id))
                  #     .group_by(Tips.type, User.yelping_since, User.user_id)
                  #     .order_by(fn.Count('*').desc())
                  #     .limit(5)
              ]
    for query in queries:

        if not hasattr(query, 'execute'):
            print query
            continue
        print list(query)
