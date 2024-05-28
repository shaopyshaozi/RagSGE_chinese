from setuptools import setup, find_packages

# Open the README file with the correct encoding
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='RagSGE_chinese',
    version='0.14',
    packages=find_packages(),
    install_requires=[
        'openai', 
        'langchain',
        'langchain-openai',
        'python-dotenv',
        'elasticsearch==8.8.2',
        'tqdm',
        'scipy',
        'tiktoken',
        'pandas',
        'datasets',
        'ragas',
        'openpyxl',
    ],
    long_description_content_type='text/markdown',  # Use 'text/x-rst' if you are using reStructuredText
    long_description=long_description,  # Ensure your README file is at the specified location
)