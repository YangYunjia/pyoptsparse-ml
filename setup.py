from setuptools import setup, find_packages
from pyoptsparse_ml import __name__, __version__

# with open('README.md') as f:
#       long_description = f.read()

setup(name=__name__,
      version=__version__,
      description='ML CFD api for pyoptsparse',
      keywords=['CFD', 'machine learning', 'optimization'],
      author='Aerolab',
      author_email='yyj980401@126.com',
      packages=find_packages(),
    #   install_requires=['numpy'],
      classifiers=[
            'Programming Language :: Python :: 3'
      ]
)

