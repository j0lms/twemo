# twemo
Twitter emotion detection using PyTorch. Built with Python 3.9.5 and Torch 1.11.0 and Tweepy 4.9.0. Based on the text classification model provided in [this example](https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html)
# Install
1. Download the requirements:
	`pip install -r requirements.txt`
2. Replace your Twitter API keys in the vars.py file:
	`access_token_secret = 'Your key goes here'`
3. Specify a directory for the dataset files:
	`csvdir = 'home/twemo/datasets'`
4. Run:
	`python /tweemo`
# Usage
1. Get tweets:
Allows you to get tweet-emotion pairs into the dataset.csv file. You can specify an arbitrary amount of emotion labels that you will train with on by modifying the `emotion_labels` variable.
2. Create/Update Dataset:
Creates the test.csv and train.csv split.
3. Train:
Trains the model on the provided test and train files.
4. Predict string:
Gives an emotion label prediction based on a string with a percentage breakdown of a number of predictions.
5. Timeline Diet:
Gets prediction labels for all tweets in a user's timeline.
