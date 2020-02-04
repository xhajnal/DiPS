from setuptools import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


setup(
    name='package name',
    # version='version',
    install_requires=requirements,
)
