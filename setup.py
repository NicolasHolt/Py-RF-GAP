from setuptools import setup

setup(
    name='PY-RF-GAP',
    version='0.1',
    packages=['PY_RF_GAP'],
    install_requires=[
        'numpy>=1.24.4',
        'scikit-learn>=1.5.0',
    ],
    author='Nicolas Holt, Luke Gehring',
    author_email='your.email@example.com',
    description='Python implementation of Geometry- and Accuracy-Preserving Random Forest Proximities: arXiv:2201.12682',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NicolasHolt/Py-RF-GAP',
)
