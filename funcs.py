import torch
import time
import os
import tweepy
import csv
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import TWEET_SET
from models import TextClassificationModel
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchtext.data.functional import to_map_style_dataset
from vars import csvdir, emotion_labels, consumer_key, consumer_secret, access_token, access_token_secret, account_list, test_string, tweet_number, EPOCHS, LR, BATCH_SIZE, iteration_number

os.chdir(csvdir)

auth = tweepy.OAuth1UserHandler(
    consumer_key, consumer_secret, access_token, access_token_secret
    )
api = tweepy.API(auth)
#tokenizer = get_tokenizer('basic_english')
tokenizer = get_tokenizer('spacy',language='en_core_web_sm')

def main():
    pass

def get_tweets(prompt):


    try:
        with open('dataset.csv', newline='') as csvfile:
            reader = csv.reader(csvfile)
            # print(len(reader))
            current = [ [ row[0], row[1] ] for row in reader]
            print('')
            print('Current rows: {}'.format(len(current)))
            print('')
    except FileNotFoundError:
        #with open('dataset.csv', 'w', newline='') as csvfile:
         #   writer = csv.writer(csvfile)
          #  writer.writerow(['emotion' , 'tweet'])
        pass



    # number_of_tweets=200
    if prompt == 'u':
        user = input('Input username: ')
        tweets = tweepy.Cursor(api.user_timeline, screen_name=user, tweet_mode='extended').items(tweet_number)
        #tweets = api.user_timeline(screen_name=user, tweet_mode="extended")
    elif prompt == 't':
        tweets = tweepy.Cursor(api.home_timeline, tweet_mode='extended').items(tweet_number)
        #tweets = api.home_timeline(tweet_mode="extended")
    elif prompt == 'p':
        q = input('Input search: ')
        tweets = tweepy.Cursor(api.search_tweets, q,result_type='popular', tweet_mode='extended').items(tweet_number)
        #tweets = api.search_tweets(q,result_type='popular', tweet_mode="extended")
    elif prompt == 'a':
        for account in account_list:
            account_tweets = tweepy.Cursor(api.user_timeline, screen_name=account, tweet_mode='extended').items(tweet_number)
            #account_tweets = api.user_timeline(screen_name=account, tweet_mode="extended")
            for tweet in account_tweets:
                print('-' * 59)
                try:
                    text = tweet.retweeted_status.full_text
                except AttributeError:  # Not a Retweet
                    text = tweet.full_text
                try:
                    if tweet.entities['media']:
                        media_check = True
                except:
                    media_check = False
                print('Current Bot: {} | Media: {} | Tweet Text: {}'.format(tweet.author.name, media_check, text))
                print('')
                print('Available emotions: ')
                print('')
                print(emotion_labels)
                print('')
                print('-' * 59)
                emotion = input('Input emotion as integer (p to pass, b to break): ')
                if emotion == 'p':
                    pass
                elif emotion == 'b':
                    break
                else:
                    print('-' * 59)
                    print('')
                    print('Saving...')
                    print('')
                    print('"{}","{}"'.format(emotion,tweet.full_text))
                    print(tokenizer(tweet.full_text))
                    print('')
                    print('-' * 59)
                    with open('dataset.csv', 'a', newline='') as csvfile:
                        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                        try:
	                        text = tweet.retweeted_status.full_text
                        except AttributeError:  # Not a Retweet
                            text = tweet.full_text
                        writer.writerow([int(emotion),str(text)])
                        #current = [ [ row[0], row[1] ] for row in reader]
                        # print('Current rows: {}'.format(len(current)))
                    print('')
                    print('')
                    print('')



    elif prompt == 's':
        print('-' * 59)
        print('Available emotions: ')
        print('')
        print(emotion_labels)
        print('')
        print('-' * 59)
        string = input('Input string: ')
        print('')
        emotion = input('Input emotion as integer: ')
        print('-' * 59)
        print('')
        print('Saving...')
        print('')
        print('"{}","{}"'.format(emotion,string))
        print(tokenizer(string))
        print('')
        print('-' * 59)
        with open('dataset.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow([int(emotion),str(string)])
            #current = [ [ row[0], row[1] ] for row in reader]
            # print('Current rows: {}'.format(len(current)))
        print('')
        print('')
        print('')

    elif prompt == 'l':
        print('-' * 59)
        print('Available emotions: ')
        print('')
        print(emotion_labels)
        print('')
        print('-' * 59)
        string = input('Input link (URL): ')
        m = re.search('\/([0-9]+)(?=[^\/]*$)', string)
        tweet = api.get_status(m.group(1), tweet_mode='extended')
        try:
            if tweet.entities['media']:
                media_check = True
        except:
            media_check = False
        print('Media {} | Full tweet text: {}'.format(media_check, tweet.full_text))
        print('')
        print('')
        emotion = input('Input emotion as integer: ')
        print('-' * 59)
        print('')
        print('Saving...')
        print('')
        print('"{}","{}"'.format(emotion,tweet.full_text))
        print(tokenizer(tweet.full_text))
        print('')
        print('-' * 59)
        with open('dataset.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
            writer.writerow([int(emotion),str(tweet.full_text)])
            #current = [ [ row[0], row[1] ] for row in reader]
            # print('Current rows: {}'.format(len(current)))
        print('')
        print('')
        print('')

    try:
        for tweet in tweets:
            print('-' * 59)
            try:
                text = tweet.retweeted_status.full_text
            except AttributeError:  # Not a Retweet
                text = tweet.full_text
            try:
                if tweet.entities['media']:
                    media_check = True
            except:
                media_check = False
            print('Author: {} | Media: {} | Tweet Text: {}'.format(tweet.author.name, media_check, text))
            print('')
            print('Available emotions: ')
            print('')
            print(emotion_labels)
            print('')
            print('')
            print('-' * 59)
            emotion = input('Input emotion as integer (p to pass, b to break): ')
            if emotion == 'p':
                pass
            elif emotion == 'b':
                break
            else:
                print('-' * 59)
                print('')
                print('Saving...')
                print('')
                print('"{}","{}"'.format(emotion,tweet.full_text))
                print(tokenizer(tweet.full_text))
                print('')
                print('-' * 59)
                with open('dataset.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL)
                    try:
                        text = tweet.retweeted_status.full_text
                    except AttributeError:  # Not a Retweet
                        text = tweet.full_text
                    writer.writerow([int(emotion),str(text)])
                    #current = [ [ row[0], row[1] ] for row in reader]
                    # print('Current rows: {}'.format(len(current)))
                print('')
                print('')
                print('')
    except UnboundLocalError:
        pass


def update_dataset():


    with open('dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        emotions = [ row[0] for row in reader]

    with open('dataset.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        tweets = [ row[1] for row in reader]

    raw_data = {'emotions':emotions,
            'tweets':tweets
            }

    #print(raw_data)

    df = pd.DataFrame(raw_data, columns=['emotions','tweets'])
    print('-' * 59)
    print('')
    print('| dataframe')
    print(df)
    print('')
    print('-' * 59)
    #df['emotions'] = df['emotions'].apply(lambda x: list(map(int, x)))

    train, test = train_test_split(df, test_size=0.2)
    print('-' * 59)
    print('')
    print('| train.csv')
    print(train)
    print('')
    print('-' * 59)
    print('-' * 59)
    print('')
    print('| test.csv')
    print(test)
    print('')
    print('-' * 59)

    #train.to_json('train.json', orient='records', lines=True)
    #test.to_json('test.json', orient='records', lines=True)

    train.to_csv('train.csv', index=False)
    test.to_csv('test.csv', index=False)

    #spacy_eng = spacy.load('en_core_web_sm')

    #def tokenize_tweet(text):
     #   return [tok.text for tok in spacy_eng.tokenizer(text)]

    #emotion = data.LabelField(sequential=True, use_vocab=False)
    #tweet = data.Field(sequential=True, use_vocab=True, tokenize=tokenize_tweet, lower=True)

    #fields = {'emotions':('e', emotion), 'tweets':('t', tweet)}

    #train_data, test_data = data.TabularDataset.splits(
     #                           path='',
     #                           train='train.csv',
     #                           test='test.csv',
     #                           format='csv',
     #                           fields=fields
      #                          )

    #tweet.build_vocab(train_data, max_size=280, min_freq=2)

    #train_iterator, test_iterator = data.BucketIterator.splits(
    #    (train_data, test_data),
     #   batch_size=2,
     #   device='cpu')

    #for batch in train_iterator:
    #    print(batch)

    return raw_data


def train():

    #data processing pipeline

    train_iter = TWEET_SET(split='train')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1

    #data batch and iterator

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def collate_batch(batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
             label_list.append(label_pipeline(_label))
             processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
             text_list.append(processed_text)
             offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return label_list.to(device), text_list.to(device), offsets.to(device)

    train_iter = TWEET_SET(split='train')
    dataloader = DataLoader(train_iter, batch_size=8, shuffle=False, collate_fn=collate_batch)

    train_iter = TWEET_SET(split='train')
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64
    model = TextClassificationModel(vocab_size, emsize, num_class).to(device)


    #train/eval functions

    def train(dataloader):
        model.train()
        total_acc, total_count = 0, 0
        log_interval = 500
        start_time = time.time()

        for idx, (label, text, offsets) in enumerate(dataloader):
            optimizer.zero_grad()
            predicted_label = model(text, offsets)
            loss = criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            total_acc += (predicted_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches '
                      '| accuracy {:8.3f}'.format(epoch, idx, len(dataloader),
                                                  total_acc/total_count))
                total_acc, total_count = 0, 0
                start_time = time.time()

    def evaluate(dataloader):
        model.eval()
        total_acc, total_count = 0, 0

        with torch.no_grad():
            for idx, (label, text, offsets) in enumerate(dataloader):
                predicted_label = model(text, offsets)
                loss = criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                total_count += label.size(0)
        return total_acc/total_count


    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)
    total_accu = None
    train_iter, test_iter = TWEET_SET()
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = \
        random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    train_dataloader = DataLoader(split_train_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(split_valid_, batch_size=BATCH_SIZE,
                                  shuffle=True, collate_fn=collate_batch)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                                 shuffle=True, collate_fn=collate_batch)

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(train_dataloader)
        accu_val = evaluate(valid_dataloader)
        if total_accu is not None and total_accu > accu_val:
          scheduler.step()
        else:
           total_accu = accu_val
        print('-' * 59)
        print('| end of epoch {:3d} | time: {:5.2f}s | '
              'valid accuracy {:8.3f} '.format(epoch,
                                               time.time() - epoch_start_time,
                                               accu_val))
        print('-' * 59)

    print('Checking the results of test dataset.')
    accu_test = evaluate(test_dataloader)
    print('test accuracy {:8.3f}'.format(accu_test))

    def predict(text, text_pipeline):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1

    model = model.to("cpu")

    print('This sparks {}. \nOriginal: {}'.format(emotion_labels[predict(test_string, text_pipeline)],test_string))


def predict(link):
    m = re.search('\/([0-9]+)(?=[^\/]*$)', link)
    text = api.get_status(m.group(1), tweet_mode='extended').full_text

    #train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter = TWEET_SET(split='train')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    train_iter = TWEET_SET(split='train')
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64

    def predict(text, text_pipeline):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1

    #model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    #model = model.to("cpu")

    pred_list = []

    for i in range(iteration_number):
        model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
        model = model.to("cpu")
        prediction = predict(text, text_pipeline)
        pred_list.append(prediction)
        label_dict = {pred_list.count(label):label for label in emotion_labels }
        highest = label_dict[sorted(label_dict)[len(label_dict)-1]]
        print('This sparks {} | {}/{}'.format(emotion_labels[prediction],i,iteration_number))


    print('-' * 59)
    print('')
    print('Text: {}'.format(text))
    print('')
    print('Likeliest emotion {}.'.format(emotion_labels[highest]))
    print('')
    for label in emotion_labels:
        print('| likelihood of {} | iterations #{} ({}%)'.format(emotion_labels[label], pred_list.count(label), (pred_list.count(label)/len(pred_list))*100))
    print('')
    print('-' * 59)

def mass_predict(prompt):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_iter = TWEET_SET(split='train')

    def yield_tokens(data_iter):
        for _, text in data_iter:
            yield tokenizer(text)

    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    text_pipeline = lambda x: vocab(tokenizer(x))

    train_iter = TWEET_SET(split='train')
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    emsize = 64

    def predict(text, text_pipeline):
        with torch.no_grad():
            text = torch.tensor(text_pipeline(text))
            output = model(text, torch.tensor([0]))
            return output.argmax(1).item() + 1

    #model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
    #model = model.to("cpu")

    # tweets = api.home_timeline(tweet_mode="extended")

    if prompt == 'u':
        user = input('Input username: ')
        tweets = tweepy.Cursor(api.user_timeline, screen_name=user, tweet_mode='extended').items(tweet_number)
    elif prompt == 't':
        tweets = tweepy.Cursor(api.home_timeline, tweet_mode='extended').items(tweet_number)
    elif prompt == 'p':
        q = input('Input search: ')
        tweets = tweepy.Cursor(api.search_tweets, q,result_type='popular', tweet_mode='extended').items(tweet_number)

    pred_list = []

    for tweet in tweets:
        tweet_list = []
        for i in range(iteration_number):
            model = TextClassificationModel(vocab_size, emsize, num_class).to(device)
            model = model.to("cpu")
            prediction = predict(tweet.full_text, text_pipeline)
            tweet_list.append(prediction)

        label_dict = {tweet_list.count(label):label for label in emotion_labels }
        highest = label_dict[sorted(label_dict)[len(label_dict)-1]]
        pred_list.append(highest)

        #for result in dict_list:
        #    count_list.append(result)

        #highest = sorted(count_list)[len(count_list)-1]
        #print(sorted(count_list))

        print('-' * 59)
        print('This sparks {}'.format(emotion_labels[highest]))
        print('-' * 59)
        print('')
        print('Text: {}'.format(tweet.full_text))
        print('')
        for label in emotion_labels:
            print('| likelihood of {} | iterations #{} ({}%)'.format(emotion_labels[label], tweet_list.count(label), (tweet_list.count(label)/len(tweet_list))*100))
        print('')
        print('-' * 59)

    print('-' * 59)
    print('')
    print('Total emotional consumption:')
    print('')
    for label in emotion_labels:
        #print('')
        print('Total {}: {} tweets ({}%)'.format(emotion_labels[label], pred_list.count(label), (pred_list.count(label)/len(pred_list))*100))
    print('')
    print('-' * 59)
