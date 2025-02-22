from setuptools import setup, find_packages

VERSION = '0.0.1.3' 
DESCRIPTION = 'My first Python package for extracting Functional Connectivity Matrices'
LONG_DESCRIPTION = 'FCM available: Correlation FCM'

# Setting up
setup(
        name="fcmtool_bqmn", 
        version=VERSION,
        author="Quoc Minh Nhat Bui",
        author_email="star.buiquocminhnhat.0510@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=['numpy', 'glob2', 'pathlib', 'nilearn', 'load_confounds'], # dependencies
        keywords=['python', 'first package'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)