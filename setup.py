from setuptools import setup

setup(
    name='downward-mtree',
    version='1.0',
    py_modules=["dmtree"],
    url='https://github.com/Napolei/dmtree',
    license='MIT License',
    author='Max Feltes',
    author_email='contact@feltes.ch',
    description='A variant of MTrees, where NonLeafNodes do not propagate splits to the parent, but combine routing objects downwards'
)
