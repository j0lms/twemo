import os

consumer_key = os.getenv('CONSUMER_KEY')
consumer_secret = os.getenv('CONSUMER_SECRET')
access_token = os.getenv('ACCESS_TOKEN')
access_token_secret = os.getenv('ACCESS_TOKEN_SECRET')

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

            # Provide user handles for the account roll
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

csvdir = 'twemo/datasets' # Provide a folder for the dataset.csv, test.csv, train.csv files

test_string = 'This is a test string.' # String to test after performing training