import time
import numpy as np
from my_model_inference import get_responses, N_RESPONSE


######################
# ReplySuggest Class #
######################
class ReplySuggest():
    def __init__(self):
        # a flag to check whether to end the conversation
        self.end_chat = False
        # greet while starting
        self.welcome()

    def welcome(self):
        print("Initializing ReplySuggest ...")
        # some time to get user ready
        time.sleep(2)
        print('Type "bye" or "quit" or "exit" to end chat \n')
        # give time to read what has been printed
        time.sleep(3)
        # Greet and introduce
        greeting = np.random.choice([
            "Hey! Nice to meet you. Please type in your message.",
            "Hello, it's my pleasure meeting you. Please type in your message.",
            "Hi, this is ReplySuggest System! Please type in your message."
        ])
        print("ReplySuggest >>  " + greeting)

    def user_input(self):
        # receive input from user
        text = input("User         >>  ")
        # end conversation if user wishes
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            # turn flag on
            self.end_chat = True
            # a closing comment
            print('ReplySuggest >>  See you soon! Bye!')
            time.sleep(1)
            print('\nQuitting ReplySuggest ...')
        else:   # continue chat
            self.input = text

    def bot_response(self):
        response = get_responses(self.input, N_RESPONSE)
        for i in range(len(response)):
            print("Suggestion[" + str(i+1) + ']: ' + response[i][0])
        print("ReplySuggest >>  If there is NO desired suggestions, RETYPE again!")
        print("                 Otherwise, type in new message.")


if __name__ == '__main__':
    bot = ReplySuggest()
    while True:
        bot.user_input()
        if bot.end_chat:
            break
        bot.bot_response()
