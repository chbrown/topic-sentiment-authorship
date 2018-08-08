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
    install_requires=open('requirements.txt').readlines(),
)
