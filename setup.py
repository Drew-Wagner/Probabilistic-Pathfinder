import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.readlines()

setuptools.setup(
    name="probabilistic-pathfinder-drew-wagner",
    version="0.1.0",
    author="Drew Wagner",
    author_email="drewarthurwagner@gmail.com",
    description="A 3D probabilistic path finder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Drew-Wagner/Probabilistic-Pathfinder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.8',
    install_requires=requirements
)
