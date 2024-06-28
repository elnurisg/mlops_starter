from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()
requirements_path = here / "requirements.txt"

# Get the long description from the README file
# long_description = (here / "README.md").read_text(encoding="utf-8")

def parse_requirements(filename):
    with open(filename, 'r') as file:
        return [line.strip() for line in file if line and not line.startswith('#')]

setup(
    name="mlops_starter",
    version="1.0.0", 
    description="A sample mlops project",  
    # long_description=long_description,
    # long_description_content_type="text/markdown", 
    packages=find_packages(where="ml_package"), 
    python_requires=">=3.7, <4",
    install_requires=parse_requirements(requirements_path)
)