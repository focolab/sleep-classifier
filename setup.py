import setuptools

setuptools.setup(
    name="sleep_scorer",
    version="0.1",
    author="Greg Bubnis",
    author_email="gregory.bubnis@ucsf.edu",
    description="Classify mouse sleep states from EEG and EMG recordings",
    long_description_content_type=open('README.md').read(),
    url="http://www.github.com/focolab/sleep-classifier",
    packages=['sleep_scorer'],
    install_requires=[
          'numpy>=1.13.3',
          'matplotlib>=2.1.0',
          'seaborn>=0.10.1',
          'pandas>=1.0.03',
          'jupyter>=1.0.0',
          'scikit-learn>=0.22.2',
          'plotly>=4.7.1',
          'pyedflib @ git+https://github.com/holgern/pyedflib@master',
      ], 
    dependency_links=[],
    python_requires='>=3.6',
)