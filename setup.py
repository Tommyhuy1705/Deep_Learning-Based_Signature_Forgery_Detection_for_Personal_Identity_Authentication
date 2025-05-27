from setuptools import setup, find_packages

# Read requirements.txt to get the list of dependencies
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='signature_verification',  # Name package
    version='0.1.0',                # Version
    packages=find_packages(),       # Automatically find all packages (with __init__.py)
    install_requires=requirements,  # Use requirements.txt
    author='Dong,Huy,Huong,Nhut,Thien',
    description='A project for signature verification using Triplet Siamese Similarity Network',
    python_requires='>=3.8',        # Minimum Python version
)