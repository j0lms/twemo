from funcs import main, update_dataset, get_tweets, train, predict, mass_predict

if __name__ == '__main__':
    def help_me():
        print('')
        print('               Twitter Emotion Detector                     ')
        print('                                                            ')
        print('               1. Get tweets                                ')
        print('               2. Create/Update Data Set                    ')
        print('               3. Train                                     ')
        print('               4. Predict String                            ')
        print('               5. Timeline Diet                             ')
        print('                                                            ')
        print('                        (q) to Exit                         ')
        print('                        (h) for Help                        ')
        print('                                                            ')
    help_me()
    while True:
        command = input('Input command: ')
        if str(command) == 'q':
            exit()
        elif str(command) == 'h':
            help_me()
        elif str(command) == '1':
            prompt = input('Fetch from the timeline (t), from an username (u), from an account list (a), from a popular search (p), add a tweet manually (s): ')
            get_tweets(prompt)
        elif str(command) == '2':
            update_dataset()
        elif str(command) == '3':
            train()
        elif str(command) == '4':
            prompt = input('Input string: ')
            predict(prompt)
        elif str(command) == '5':
            mass_predict()
    main()