# text-categorization
Text categorization/classification project. e-commerce. KHPI diploma

## Prerequisites:
1. clone repository:
```
git clone https://github.com/MikeHoangg/text-categorization.git
```
2. create virtual environment, activate it:
```
virtualenv venv
. venv/bin/activate
```
3. install requirements:
```
pip install -r requirements
```
4. install packages data:
```
python -m spacy download en
```

## Usage:
1. Create pipeline config using example - [ref](https://github.com/MikeHoangg/text-categorization/blob/master/config.yaml)
2. Use `ProductTextProcessor` class from [this](https://github.com/MikeHoangg/text-categorization/blob/master/src/base.py) module for running your pipeline