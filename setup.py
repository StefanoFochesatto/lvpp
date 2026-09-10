from setuptools import setup

setup(
    name='lvpp',
    version='0.1',
    author='Stefano Fochesatto',
    description='Latent variable proximal point for variational inequalities',
    long_description='Latent variable proximal point (LVPP) algorithm for variational '
                     'inequalities, after Dokken, Farrell, Keith, Papadopoulos, Surowiec, '
                     'arXiv:2503.05672.',
    packages=['lvpp'],
    zip_safe=False,
)
