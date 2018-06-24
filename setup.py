import json
import setuptools

package = json.load(open('package.json'))

setuptools.setup(
    name=str(package['name']),
    version=str(package['version']),
    url=str(package['homepage']),
    author=str(package['author']['name']),
    author_email=str(package['author']['email']),
    license=open('LICENSE').read(),
    packages=setuptools.find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'gensim == 0.8.6',
        'justext == 2.0.0',
        'openpyxl == 1.6.2',
        'psycopg2 == 2.5.1',
        'requests == 1.2.3',
        'sqlalchemy == 0.8.2',
        'scikit-learn == 0.14.1',
    ],
    entry_points={
        'console_scripts': [
            'tsa-hashtags=tsa.cli.hashtags:main',
            'tsa-analysis=tsa.cli.analysis:main',
        ],
    },
)
