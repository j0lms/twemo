# twemo
Twitter emotion detection using PyTorch. Built with Python 3.9.5 and Torch 1.11.0. Based on the text classification model provided in [this example](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
# Install
1. Creating the Python environment:
	`python -m venv twemo-venv`
2. Download the requirements:
	`pip install -r requirements.txt`
3. To enable the .env file, create a file called postactivate in the Scripts folder of your environment and add this line, changing the directory as appropriate (You can also skip this step by passing the tokens into the variables in vars.py as strings):
	`set -a; source ~/twemo/.env; set +a`
4. Replace your Twitter API keys in the .env file:
	`export ACCESS_TOKEN_SECRET='Your key goes here'`
5. Run:
	`python /tweemo`
# Usage
1. Get tweets:
Allows you to get tweet-emotion pairs into the dataset.csv file
2. Create/Update Dataset:
Creates the test.csv and train.csv split
3. Train:
Trains the model on the provided test and train files (URL only)
4. Predict string:
Gives an emotion label prediction based on a string
5. Timeline Diet:
Gets predictions for all tweets in a timeline
