# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='ad_nids',
    version='0.1.0',
    description='Are deep generative models suitable for use in anomaly-based network intrusion detection systems?',
    long_description=readme,
    author='Mikhail Mishin',
    author_email='mishinma1805@gmail.com',
    url='https://github.com/mishinma/ad-nids',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

