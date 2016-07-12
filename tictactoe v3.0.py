# -*- coding: utf-8 -*-
"""
I applied the TD(lambda) learning method to teach an agent how to play at 
TicTacToe (TTT). The agent learns by playing against itself.

As TTT is a deterministic and memoryless game, the agent only needs to work 
out a state-value function rather than a state/action-value function. In first 
approximation, the maximum number of states is capped at 6046, hence the 
number of possible states is small enough to be stored and valued.

I applied the updating formula
    Q(s) <- Q(s) + ALPHA * [ Reward + GAMMA*Q(S_t+1) - Q(S_t)] * Tr(s)

The agent tries to load a state-value function from file. If it can't find it 
in the folder, it will work it out from scratch. It takes a few minutes to 
derive a decent enough function.

Created on Sun May 01 12:11:56 2016
@author: Riccardo Rossi
"""

# Import libraries
import sys
import csv
import numpy as np
import pygame
import time
np.random.seed(int(time.time()))
import menu                         # menu.py should be in the same folder

### Learning Parameters
WIN_VALUE = 1.
TIE_VALUE = 0.
# Set the initial valua higher than TIE_VALUE to incentivise exploration
INITIAL_STATE_VALUE = TIE_VALUE + (WIN_VALUE - TIE_VALUE) / 5.  

EPSILON = 0.1   # starting probability of exploration choice
ALPHA = 0.1      # learning rate
GAMMA = 0.95      # discounting factor
LAMBDA_ = 0.3   # hyperparameter of the learning method

MAX_EPOCHS = int(5e4)
FREQUENCY_SHOW = int(1e4)

CODE_PLAYER_1 = 'X'
CODE_PLAYER_2 = '0'
CODE_WINNER = 'W'
GAME_TITLE = 'Tic Tac Toe'

start = time.time()
##########################################################################


class Board():

    def __init__(self):
        self._reboot()
        self.state_value_function = {self.state: TIE_VALUE}
        self.td_traces = {self.state: 0.}

    def _reboot(self):
        # Reset the TTT board
        self.state = ''.join(['-' for _ in range(0, 9)])
        self.game = []

    def decide(self, greedy=False):
        # Agent decides the next move

        # Identify player: +1 is player1 and -1 is player2
        who_is_playing = 1 - 2 * ((9 - self.state.count('-')) % 2)
        character = CODE_PLAYER_1 if who_is_playing > 0 else CODE_PLAYER_2

        # Work out the list of next possible boards (starting from the current
        # state)
        possible_moves = []
        for i in range(0, 9):
            if self.state[i] == '-':
                elem = list(self.state)
                elem[i] = character
                possible_moves.append(''.join(elem))

        # Assign default value to states visited for the first time
        for elem in possible_moves:
            if not elem in self.state_value_function.keys():
                self.state_value_function[elem] = INITIAL_STATE_VALUE
                self.td_traces[elem] = 0.

        # Choose the next move, called decision, which is in the format of
        # (next) board state
        optimal_moves = []
        if greedy or np.random.random() > EPSILON:
            max_value = max(self.state_value_function[elem] 
                            for elem in possible_moves)
            for elem in possible_moves:
                if not self.state_value_function[elem] < max_value:
                    optimal_moves.append(elem)
        else:
            optimal_moves = possible_moves
        decision = optimal_moves[np.random.randint(0, len(optimal_moves))]

        # Update the board state with the new decision taken
        self.state = decision

        # Add the move into the history of the game
        self.game.append(self.state)

    def _check_end_and_update_state_values(self, show=False):

        # Add the current state in case it's not in memory
        if not board.state in self.state_value_function.keys():
            self.state_value_function[self.state] = INITIAL_STATE_VALUE
            self.td_traces[self.state] = 0.

        # Check if there is a winner
        there_is_a_winner = False
        has_game_ended = False
        reward = 0
        for i in range(0, 3):
            if self.state[3*i] == self.state[3*i+1] == self.state[3*i+2] !='-':
                there_is_a_winner = True
                elem = list(self.state)
                elem[3*i] = elem[3*i+1] = elem[3*i+2] = CODE_WINNER
                self.state = ''.join(elem)
            if self.state[i] == self.state[i+3] == self.state[i+6] != '-':
                there_is_a_winner = True
                elem = list(self.state)
                elem[i] = elem[i+3] = elem[i+6] = CODE_WINNER
                self.state = ''.join(elem)
        if self.state[0] == self.state[4] == self.state[8] != '-':
            there_is_a_winner = True
            elem = list(self.state)
            elem[0] = elem[4] = elem[8] = CODE_WINNER
            self.state = ''.join(elem)
        if self.state[2] == self.state[4] == self.state[6] != '-':
            there_is_a_winner = True
            elem = list(self.state)
            elem[2] = elem[4] = elem[6] = CODE_WINNER
            self.state = ''.join(elem)

        if there_is_a_winner:
            reward = WIN_VALUE
            has_game_ended = True
        elif '-' not in self.state:
            reward = TIE_VALUE
            has_game_ended = True

        ## Assign rewards via TD(lambda) method #############################
        # Note I choose to do this only at the end of the game rather than
        # after every move as it should be
        if has_game_ended:
            for i in range(2, len(self.game) + 2):
                self.td_traces[self.game[i - 2]] = 1.

                # Calculate the fixed Temporal Difference
                if i >= len(self.game):
                    TD = (reward * (-1)**(1 + len(self.game) - i) 
                        - self.state_value_function[self.game[i-2]])
                else:
                    TD = (GAMMA * self.state_value_function[self.game[i]] 
                        - self.state_value_function[self.game[i-2]])

                # Update the state_value with the Temporal Difference
                for state in self.game:
                    self.state_value_function[state] += ALPHA * (
                                    TD * self.td_traces[state])
                    # Update the trace dictionary. Note the neg sign that
                    # allows for opposite updates for the opposing player
                    self.td_traces[state] *= -LAMBDA_ * GAMMA

            # Set the eligibility to zero at the end of the episode
            for state in self.game:
                self.td_traces[state] = 0.
        ######################################################################

        if show:
            self._show()
        return(has_game_ended)

    def _show(self):
        # Dispaly the board in text version

        for i in range(0, 3):
            print (self.state[3*i : 3*i + 3])
            
        if CODE_WINNER in self.state:
            state_value = WIN_VALUE 
        else:
            state_value = self.state_value_function[self.state]
            
        print ('Prob of winning for player:', state_value)
        sys.stdout.flush()
        print(' ')


class Player():

    def __init__(self, code):
        self.code = code

board = Board()

# Attempt to download the Agent's parameters
try:
    with open('TicTacToe_parameters', 'r') as file_data:
        board.state_value_function = dict(csv.reader(file_data))
        for key in board.state_value_function.keys():
            board.state_value_function[key] = float(
                board.state_value_function[key])
            board.td_traces[key] = 0.
        file_data.close()
    is_training = False
except IOError:
    is_training = True
    
# Agent goes through training as it could not load parameters
if is_training:    
    player1 = Player(CODE_PLAYER_1)
    player2 = Player(CODE_PLAYER_2)
    print ('Working out state-value function ...')
    sys.stdout.flush()
    for epoch in range(0, MAX_EPOCHS):
        show = True if epoch % FREQUENCY_SHOW == 0 else False
        if show:
            print('-----------------------\n')
        while not board._check_end_and_update_state_values(show):
            board.decide()
        board._reboot()


def screen_display(Human, AI):
    # Display the current board via Pygame

    screen.fill(WHITE)
    pygame.draw.line(screen, (0, 0, 0), (100, 300), (100, 0))
    pygame.draw.line(screen, (0, 0, 0), (200, 300), (200, 0))
    pygame.draw.line(screen, (0, 0, 0), (300, 100), (0, 100))
    pygame.draw.line(screen, (0, 0, 0), (300, 200), (0, 200))
    for i in range(0, len(board.state)):
        if board.state[i] == AI.code:
            pygame.draw.rect(screen, BLUE, (int(i/3)*100, 
                                            (i%3)*100, 100, 100))
        if board.state[i] == Human.code:
            pygame.draw.rect(screen, RED, (int(i/3)*100, 
                                           (i%3)*100, 100, 100))
        if board.state[i] == CODE_WINNER:
            pygame.draw.rect(screen, YELLOW, (int(i/3)*100, 
                                              (i%3)*100, 100, 100))
    pygame.display.flip()


BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 192, 203)
BLUE = (176, 196, 222)
YELLOW = (218, 165, 32)

# Initialize Pygame
pygame.init()
pygame.display.set_caption(GAME_TITLE)
screen = pygame.display.set_mode((300, 300))

# Launch the menu and receives instructions over which player starts first
HUMAN_STARTS = menu.launch(screen)

# Launch the game
running = True
while running:

    # AI makes the first move if it starts first
    if not HUMAN_STARTS:
        AI = Player(CODE_PLAYER_1)
        Human = Player(CODE_PLAYER_2)
        board.decide(greedy=True)
    else:
        AI = Player(CODE_PLAYER_2)
        Human = Player(CODE_PLAYER_1)

    # Update the screen
    screen_display(Human, AI)

    # Main game cycle
    game_on = True
    while game_on:

        # Check if a quit command is given
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = game_on = False
                continue

            # Receive the Human's decision
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = event.pos
                decision = (int(mouse_position[0]/100)*3 
                            + int(mouse_position[1]/100))
                if list(board.state)[decision] == '-':
                    temp = list(board.state)
                    temp[decision] = Human.code
                    board.state = ''.join(temp)
                    board.game.append(board.state)

                    # Check if there's a winner
                    if board._check_end_and_update_state_values():
                        game_on = False

                    # AI makes a decision
                    if game_on:
                        board.decide(greedy=True)
                        if board._check_end_and_update_state_values():
                            game_on = False

        # Display scren
        screen_display(Human, AI)

    screen_display(Human, AI)
    time.sleep(1)
    board._reboot()

# Close PyGame
pygame.display.quit()
pygame.quit()

# Save the Agent's parameters
if is_training:
    with open('TicTacToe_parameters', 'w') as file_data:
        writer = csv.writer(file_data)
        for key, value in board.state_value_function.items():
            writer.writerow([key, value])
        file_data.close()


##########################################################################
print("The process took : ", round(time.time() - start, 2), ' seconds')
