from setuptools import find_packages, setup

classes = """
    Development Status :: 5 - Production/Stable
    License :: OSI Approved :: BSD License
    Topic :: Software Development :: Libraries
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Bio-Informatics
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3 :: Only
    Programming Language :: Python :: 3.4
    Programming Language :: Python :: 3.5
    Programming Language :: Python :: 3.6
    Programming Language :: Python :: 3.7
    Operating System :: Unix
    Operating System :: POSIX
    Operating System :: MacOS :: MacOS X
"""
classifiers = [s.strip() for s in classes.split('\n') if s]

description = ('Structural similarity.')
extras = {
    'dev': [
        'wandb',
        'matplotlib',
        'pre-commit',
        'pandas',
    ],
    'deepblast': ['deepblast']
}

setup(name='tm-vec',
      version='1.0.4',
      license='BSD-3-Clause',
      description=description,
      author_email="tymorhamamsy@gmail.com",
      maintainer_email="tymorhamamsy@gmail.com",
      packages=find_packages(),
      install_requires=[
          'numpy', 'torch', 'lightning', 'pysam', 'transformers',
          'SentencePiece', 'faiss-cpu'
      ],
      extras_require=extras,
      scripts=[
          'scripts/build_database.py',
          'scripts/tmvec-search',
      ],
      classifiers=classifiers,
      package_data={})
