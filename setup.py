from distutils.core import setup

setup(
    name="neuralode",
    version="0.1dev",
    packages=["neuralode", "rl"],
    install_requires=[
        'numpy',
        'opencv-python',
        'gym[atari,mujoco]>=0.11',
        'tqdm',
    ],
    license='MIT',
    long_description=''
)
