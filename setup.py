from setuptools import setup, find_packages

setup(name='fastscode',
      version='0.0.8',
      description='FastSCODE',
      # url='http://github.com/cxinsys/fasttenet',
      author='Complex Intelligent Systems Laboratory (CISLAB)',
      author_email='rakbin007@naver.com',
      license='BSD-3-Clause',
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=find_packages(),
      install_requires=['numpy', 'statsmodels', 'networkx', 'tqdm', 'matplotlib', 'omegaconf', 'fasttenet', 'scikit-learn'],
      zip_safe=False,)
