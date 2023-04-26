from setuptools import find_packages, setup


def read_requirements(filename: str):
    with open(filename) as requirements_file:
        requirements = []
        for line in requirements_file:
            line = line.strip()
            if line.startswith("#") or len(line) == 0:
                continue
            requirements.append(line)
    return requirements


setup(
    name="fourcastnet",
    version="0.1.0",
    packages=["fourcastnet"],
    install_requires=read_requirements("requirements.txt")
)