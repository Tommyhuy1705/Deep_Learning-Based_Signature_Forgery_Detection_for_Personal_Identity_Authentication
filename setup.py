from setuptools import setup, find_packages

# Đọc requirements.txt để lấy danh sách phụ thuộc
with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='signature_verification',  # Tên package
    version='0.1.0',               # Phiên bản
    packages=find_packages(),      # Tự động tìm tất cả các package (có __init__.py)
    install_requires=requirements, # Sử dụng requirements.txt
    author='Dong,Huy,Huong,Nhut,Thien',
    author_email='xxx.@gmail.com',
    description='A project for signature verification using Siamese/Triplet networks',
    python_requires='>=3.8',       # Phiên bản Python tối thiểu
)