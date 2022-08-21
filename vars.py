consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_token_secret = 'ACCESS_TOKEN_SECRET'

emotion_labels = {

                1: 'Serenity',
                2: 'Annoyance',
                3: 'Curiosity',
                4: 'Joy',
                5: 'Awe',
                6: 'Surprise',
                7: 'Fear',
                8: 'Sadness',
                9: 'Outrage',

            }

account_list = [

            'SylviaPlathBot',
            'DailyKerouac',
            'carsonbot',
            'williamblakebot',
            'AllWittgenstein',
            'Rumi_Quote',
            'BashoSociety',
            'botvirginia',
            'cavafybot',
            'marxhaunting',
            'Scruton_Quotes',
            'TheMarkTwain',
            'Kurt_Vonnegut',
            'DebsEbooks',
            'hbloomquotes',
            'TAMinusContext',
            'k_punk_unlife',

        ]

csvdir = '/datasets' # This is where all the traning data goes

test_string = 'This is a test string'

tweet_number = 10
iteration_number = 500

# Hyperparameters
EPOCHS = 10 # epoch
LR = 5  # learning rate
BATCH_SIZE = 64 # batch size for training