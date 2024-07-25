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
    name="fme",
    version="0.1.0",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    url="https://github.com/ai2cm/ace",
    author="Allen Institute for Artificial Intelligence",
    author_email="oliverwm@allenai.org",
    license="Apache",
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"],
    ),
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": read_requirements("dev-requirements.txt"),
        "docs": read_requirements("docs/requirements.txt"),
    },
    python_requires=">=3.8",
)
