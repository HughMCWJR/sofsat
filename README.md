# SOFSAT Site
Website for Auburn's SOFSAT research project. Click [here](https://sofsat.herokuapp.com) to see the live version of the site.

To run the website locally, follow these instructions:

1. Clone or download the repository.

2. (Optional) Create and activate a Python virtual environment.
   - This step is not required but is highly recommended. If unfamiliar with Python virtual environments, follow the steps [here](https://packaging.python.org/guides/installing-using-pip-and-virtual-environments/) to get started.

3. Install dependencies.
   - Run the following line in your terminal to install the dependencies required.
```
$ pip install -r requirements.txt
```

4. Install required NLTK modules.
   - This project uses data from the Natural Language Training Kit. More can be learned about NLTK [here](https://www.nltk.org), and the data that we use can be downloaded by running the following in the terminal.
```
$ python -m nltk.downloader stopwords
$ python -m nltk.downloader punkt
```

5. Target application as the Flask app
   - Run the following in the terminal to notify Flask which file to run. This must be done everytime the terminal is re-opened.
```
$ set FLASK_ENV=development
$ set FLASK_APP=app.py
$ flask run
```
or
```
$ $env:FLASK_ENV="development"
$ $env:FLASK_APP="app.py"
$ flask run
```
6. Run the Flask app.
   - Run the folling in the terminal to run the Flask app. Running this will output a local location where the app will be running and can be viewed any browser on the same computer. Learn more about Flask [here](https://flask.palletsprojects.com/en/1.1.x/).
 
```
$ flask run
```
