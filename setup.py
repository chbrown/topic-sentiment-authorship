from setuptools import setup, find_packages

import os
import json

here = os.path.dirname(__file__) or os.curdir
package = json.load(open(os.path.join(here, 'package.json')))

setup(
    name=str(package['name']),
    version=str(package['version']),
    url=str(package['homepage']),
    author=str(package['author']['name']),
    author_email=str(package['author']['email']),
    license=open(os.path.join(here, 'LICENSE')).read(),
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'gensim >= 0.8.6',
        'justext >= 2.0.0',
        'openpyxl >= 1.6.2',
        'psycopg2 >= 2.5.1',
        'requests >= 1.2.3',
        'sqlalchemy >= 0.8.2',
        'scikit-learn >= 0.14.1',
    ],
    entry_points={
        'console_scripts': [
            'hashtags=tsa.cli.hashtags:main'
        ],
    },
)
