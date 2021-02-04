import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setuptools.setup(
    name='pymlearn',
    version='0.0.1',
    author='Jack S. Edmonds',
    author_email='jack.edmonds@me.com',
    description='Python machine learning',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/jsedmonds/pymlearn',
    packages=setuptools.find_packages(),
    install_requires=[
        'autograd',
        'numpy',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)
