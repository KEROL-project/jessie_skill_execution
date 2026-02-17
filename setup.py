import os
from setuptools import setup
from glob import glob

package_name = 'jessie_skill_execution'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml'))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Alex Mitrevski',
    maintainer_email='alemitr@chalmers.se',
    description='Package for managing skill execution on the Jessie robot',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'octo_executor = jessie_skill_execution.octo_executor:main',
            'openvla_executor = jessie_skill_execution.openvla_executor:main',
            'skill_precondition_checker = jessie_skill_execution.skill_precondition_checker:main'
        ],
    },
)
