from setuptools import setup

setup(name='spikesorting_tsne',
      version='1.0.0',
      description='T-sne with burnes hut and cuda extension (with python wrappers and python code for for spike sorting)',
      url='https://github.com/georgedimitriadis/spikesorting_tsne',
      author='George Dimitriadis',
      author_email='gdimitri@hotmail.com',
      license='MIT',
      packages=['spikesorting_tsne'],
      classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
      zip_safe=False)