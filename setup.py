import json
from setuptools import setup

package = json.load(open('package.json'))

setup(
    name=str(package['name']),
    version=str(package['version']),
    url=str(package['homepage']),
    author=str(package['author']['name']),
    author_email=str(package['author']['email']),
    license=open('LICENSE').read(),
    packages=['tsa'],
    include_package_data=True,
    zip_safe=False,
    install_requires=open('requirements.txt').readlines(),
)
