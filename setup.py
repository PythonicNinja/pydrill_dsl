#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup, find_packages
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'peewee',
]

test_requirements = [
    'peewee',
]

setup(
    name='pydrill_dsl',
    version='0.0.1',
    description="Pythonic DSL for Apache Drill",
    long_description=readme + '\n\n' + history,
    author="Wojciech Nowak",
    author_email='mail@pythonic.ninja',
    url='https://github.com/PythonicNinja/pydrill_dsl',
    packages=[
        'pydrill_dsl',
    ],
    package_dir=find_packages(
        where='.',
        exclude=('test_*', )
    ),
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='pydrill_dsl',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements
)
