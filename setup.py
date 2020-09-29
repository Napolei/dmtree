from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='downward-mtree',
    version='1.0.1',
    py_modules=["dmtree"],
    url='https://github.com/Napolei/dmtree',
    license='MIT License',
    author='Max Feltes',
    author_email='contact@feltes.ch',
    description='A variant of MTrees, where NonLeafNodes do not propagate splits to the parent, but combine routing objects downwards',
    long_description=long_description,
    long_description_content_type="text/markdown"
)
