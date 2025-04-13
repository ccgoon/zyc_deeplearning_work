from setuptools import setup, find_packages

setup(
    name='cifar10_nn',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'scikit-learn>=0.24.0',
        'tqdm>=4.50.0'
    ],
    entry_points={
        'console_scripts': [
            'cifar10-train=cifar10_nn.scripts.train:main',
            'cifar10-test=cifar10_nn.scripts.test:main'
        ]
    }
)