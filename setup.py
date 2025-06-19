import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="genoptimizer",
    version="0.1.0",
    author="Zhiqi Bu, Shiyun Xu",
    author_email="shiyunxulara@gmail.com",
    description="Pytorch implementation of Generalized Newton's method (GeN), a learning-rate-free and Hessian-informed optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ShiyunXu/AutoGeN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
