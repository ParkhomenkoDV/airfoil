from setuptools import setup, find_packages
import time

name = 'airfoil'

with open('README.md', 'rt', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', 'rt') as file:
    install_requires = file.readlines()
    print(install_requires)

setup(
    name=name,
    version=time.strftime('%Y.%m.%d.%H.%M.%S', time.localtime()),
    description='lib',
    long_description=long_description,
    long_description_content_type='text/markdown',  # если long_description = .md
    author='Daniil Andryushin',
    author_email='',
    url='https://github.com/ParkhomenkoDV/airfoil.git',
    packages=[name],
    python_requires='>=3.8',
    install_requires=install_requires,
     package_data={name:['curves.py'], }
)