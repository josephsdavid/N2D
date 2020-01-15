import setuptools
def readme():
    with open('README.md') as readme_file:
        return readme_file.read()


setuptools.setup(
    name = "n2d",
    version = "0.2.5",
    description = "(Not too) deep clustering",
    long_description = readme(),
    long_description_content_type = "text/markdown",
    url = "https://github.com/josephsdavid/N2D-OOP",
    maintainer = "david Josephs",
    maintainer_email = "josephsd@smu.edu",
   #$ packages = setuptools.find_packages(exclude = [
    #    "*weights*", "*viz*", "*data*"
   # ]),
    packages = ['n2d'],
    license = 'MIT',
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires = [
        'numpy',
        'scikit-learn',
        'umap-learn',
        'tensorflow',
        'scipy',
        'h5py >= 2.0.0',
        'matplotlib',
        'seaborn',
        'pandas',
    ],
)
