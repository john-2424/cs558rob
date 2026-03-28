from setuptools import setup

setup(
    name='modified_gym_env',
    version='0.0.1',
    install_requires=['gym==0.21.0', 
                      # 'pybullet==3.2.1'
                      'pybullet>=3.2.1,<3.3'],
    author="Jacob Johnson",
    author_email="jjj025@ucsd.edu"
)
