from setuptools import setup, find_namespace_packages
setup(
    name="Isaaclab_Parkour",
    version="0.1",    
    packages=find_namespace_packages(include=["parkour_isaaclab*", "parkour_test*"]),
)
