from setuptools import find_packages, setup

setup(
    name='ml_project',
    version='1.0.0',
    description='Package for classification task',
    author='Egor Yunusov',
    python_requires='>=3.8',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pandas>=1.2.4,<1.3'
        'numpy>=1.20.2,<1.21',
        'scikit-learn>=0.24.2,<0.25',
        'PyYAML>=5.4.1,<5.5',
        'marshmallow>=3.11.1',
        'marshmallow-dataclass>=8.4.1',
        'click>=7.1.2,<7.3',
        'Faker>=8.1.2,<8.2',
        'flake8>=3.9.1'
    ],
    extras_require={
        'test': [
            'pytest>=6.2.4,<6.3',
            'pytest-cov>=2.11.1,<3.0.0',
        ],
        'lint': [
            'mypy>=0.812,<1.0',
        ],
    },
)
