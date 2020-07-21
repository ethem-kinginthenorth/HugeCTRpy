import setuptools

setuptools.setup(
    name="HugeCTRpy",
    version="0.0.1",
    author="Ethem Can",
    author_email="ecan@nvidia.com",
    description="A Python wrapper for the HugeCTR package",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
)
