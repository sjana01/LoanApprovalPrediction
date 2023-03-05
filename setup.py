from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    '''
    This function takes a file path as input and returns a list that contain all the packages in the file
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if "-e ." in requirements:
            requirements.remove("-e .")

    return requirements

setup(
    name='mlproject',
    version='0.1',
    author='Santanil Jana',
    author_email='santanil.jana@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)