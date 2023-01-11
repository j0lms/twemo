consumer_key = 'CONSUMER_KEY'
consumer_secret = 'CONSUMER_SECRET'
access_token = 'ACCESS_TOKEN'
access_token_secret = 'ACCESS_TOKEN_SECRET'

emotion_labels = {
            1:    'Nothing',
            2:   'Annoyance',
            3:    'Curiosity',
            4:    'Joy',
            5:    'Admiration',
            6:    'Surprise',
            7:    'Fear',
            8:    'Sadness',
            9:    'Anger',
            10:    'Amusement',
            11:    'Approval',
            12:    'Caring',
            13:    'Confusion',
            14:    'Desire',
            15:    'Disappointment',
            16:    'Disapproval',
            17:    'Disgust',
            18:    'Embarrassment',
            19:    'Excitement',
            20:    'Gratitude',
            21:    'Grief',
            22:    'Love',
            23:    'Nervousness',
            24:    'Optimism',
            25:    'Pride',
            26:    'Realization',
            27:    'Relief',
            28:    'Remorse',
            29:    'Enthusiasm',
            30:    'Worry',
            31:    'Fun',
            32:    'Hate',
            33:    'Happiness',
            34:    'Boredom',

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
