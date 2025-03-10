from setuptools import find_packages, setup

setup(
    name='credit_card_risk_model',
    packages=find_packages(),
    version='0.1.0',
    description='A machine learning model for credit risk assessment',
    author='Your Name',
    license='MIT',
    install_requires=[
        # Core ML Dependencies
        'tensorflow>=2.15.0',
        'scikit-learn>=1.3.2',
        'pandas>=2.1.3',
        'numpy>=1.24.3',
        'keras-tuner>=1.4.6',
        'silence-tensorflow>=1.2.1',
        'mlflow>=2.9.0',
        
        # Utilities
        'python-dotenv>=1.0.0',
        'requests>=2.31.0',
        'pydantic>=2.5.2',
    ],
    extras_require={
        'api': [
            'flask>=3.0.0',
            'flask-cors>=4.0.0',
            'flask-limiter>=3.5.0',
            'gunicorn>=21.2.0',
            'prometheus-client>=0.19.0',
            'pyjwt>=2.8.0',
        ],
        'viz': [
            'matplotlib>=3.8.2',
            'seaborn>=0.13.0',
            'plotly>=5.18.0',
        ],
        'dev': [
            'pytest>=7.4.3',
            'pytest-cov>=4.1.0',
            'black>=23.11.0',
            'flake8>=6.1.0',
            'isort>=5.12.0',
            'mypy>=1.7.1',
            'jupyter>=1.0.0',
            'jupyterlab>=4.0.9',
        ],
        'docs': [
            'sphinx>=7.2.6',
            'sphinx-rtd-theme>=1.3.0',
            'nbsphinx>=0.9.3',
        ],
    },
    python_requires='>=3.9',
) 