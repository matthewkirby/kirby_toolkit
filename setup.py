from setuptools import setup, find_packages

setup(
      name='kirby_toolkit',
      version='1.0',
      author='Matthew Kirby',
      author_email='matthew.ryan.kirby@gmail.com',
      url='https://github.com/matthewkirby/kirby_toolkit',
      packages=find_packages(),
      description='A bunch of generic functions and classes for personal use',
      long_description=open("README.md").read(),
      package_data={"": ["README.md", "LICENSE"]},
      include_package_data=True,
      classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python"
        ],
      install_requires=["astropy", "matplotlib", "numpy", "scipy", "colossus"]
)
