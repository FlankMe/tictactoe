# -*- coding: utf-8 -*-
"""
I applied the TD(lambda) learning method to teach an agent how to play at TicTacToe (TTT).
The agent learns by playing against itself. 

As TTT is a deterministic and memoryless game, the agent only needs to work out a state-value function 
rather than a state/action-value function. In first approximation, the maximum number of states is 
capped at 6046, hence the number of possible states is small enough to be stored and valued.  

I applied the updating formula 
    Q(s) <- Q(s) + alpha * [ Reward + gamma*Q(S_t+1) - Q(S_t)] * Tr(s)

The agent tries to load a state-value function from file. If it can't find it in the folder, it 
will work it out from scratch. It takes a few minutes to derive a decent enough function. 

Created on Sun May 01 12:11:56 2016
@author: Riccardo Rossi
"""

## Import libraries
import pygame, time, csv, sys
import numpy as np
np.random.seed(int(time.time()))
import menu

# Learning Parameters
win_value = 1.
tie_value = 0.
initial_value_state = tie_value + (win_value - tie_value)/5.  # sets the initial valua higher than tie_value to incentivise exploration
epsilon = 0.1   # starting probability of exploration choice 
alpha = 0.1      # learning rate
gamma = 0.95      # discounting factor
lambda_ = 0.3   # hyperparameter of the learning method

max_epochs =     int(5e4)
frequency_show = int(1e4)

code_player1 = 'X'
code_player2 = '0'
code_winner = 'W'

game_title = 'Tic Tac Toe'
start = time.time()
#############################################################################################

class Board():
    def __init__(self):
        self.reboot()
        self.state_value_function = {self.state : tie_value}
        self.td_traces = {self.state : 0.}
            
    def reboot(self):
        # resets the TTT board
        self.state = ''.join(['-' for _ in range(0,9)])
        self.game = []
    
    def decide(self, greedy = False):     
        # Agent decides the next move
    
        # identify player: +1 is player1 and -1 is player2        
        who_is_playing =  1 - 2*( (9-self.state.count('-')) % 2 ) 
        character = code_player1 if who_is_playing > 0 else code_player2
        
        # work out the list of next possible boards (starting from the current state)        
        possible_moves = []
        for i in range(0,9):
            if self.state[i] == '-':
                elem = list(self.state)
                elem[i] = character
                possible_moves.append(''.join(elem))
        
        # assigns default value to states visited for the first time
        for elem in possible_moves: 
            if not elem in self.state_value_function.keys(): 
                self.state_value_function[elem] = initial_value_state
                self.td_traces[elem] = 0.
        
        # chooses the next move, called decision, which is in the format of (next) board state
        optimal_moves = []
        if greedy or np.random.random() > epsilon:
            max_value = max(self.state_value_function[elem] for elem in possible_moves)
            for elem in possible_moves: 
                if not self.state_value_function[elem] < max_value:
                    optimal_moves.append(elem)
        else:
            optimal_moves = possible_moves
        decision = optimal_moves[np.random.randint(0,len(optimal_moves))]
        
        # updates the board state with the new decision taken
        self.state = decision
        
        # adds the move into the history of the game
        self.game.append(self.state)
    
    def check_end_and_update_state_values(self, show=False):
        
        # add the current state in case it's not in memory
        if not board.state in self.state_value_function.keys(): 
            self.state_value_function[self.state] = initial_value_state
            self.td_traces[self.state] = 0.        
        
        # checks if there is a winner
        there_is_a_winner = False; has_game_ended = False; reward = 0
        for i in range(0,3):
            if self.state[3*i] == self.state[3*i+1] == self.state[3*i+2] != '-':
                there_is_a_winner = True
                elem = list(self.state); elem[3*i] = elem[3*i+1] = elem[3*i+2] = code_winner; self.state = ''.join(elem);
            if self.state[i] == self.state[i+3] == self.state[i+6] != '-':
                there_is_a_winner = True
                elem = list(self.state); elem[i] = elem[i+3] = elem[i+6] = code_winner; self.state = ''.join(elem);
        if self.state[0] == self.state[4] == self.state[8] != '-':
            there_is_a_winner = True    
            elem = list(self.state); elem[0] = elem[4] = elem[8] = code_winner; self.state = ''.join(elem);
        if self.state[2] == self.state[4] == self.state[6] != '-':
            there_is_a_winner = True    
            elem = list(self.state); elem[2] = elem[4] = elem[6] = code_winner; self.state = ''.join(elem);
        
        if there_is_a_winner:
            reward = win_value
            has_game_ended = True
        elif '-' not in self.state:
            reward = tie_value
            has_game_ended = True
        
        ## assigns rewards via TD(lambda) method ####################################    
        ## Note I choose to do this only at the end of the game rather than after every move as it should be
        if has_game_ended:
            for i in range(2,len(self.game)+2):
                self.td_traces[self.game[i-2]] = 1.

                # calculates the fixed Temporal Difference
                if i >= len(self.game):
                    TD = reward*(-1)**(1+len(self.game)-i) - self.state_value_function[self.game[i-2]]
                else: 
                    TD = gamma*self.state_value_function[self.game[i]] - self.state_value_function[self.game[i-2]]

                # updates the state_value with the Temporal Difference                    
                for state in self.game:
                    self.state_value_function[state] += alpha * TD * self.td_traces[state]
                    # update the trace dictionary. Note the neg sign that allows for opposite updates for the opposing player
                    self.td_traces[state] *= -lambda_ * gamma
                    
            # set the eligibility to zero at the end of the episode  
            for state in self.game:
                self.td_traces[state] = 0.
        ##############################################################################
        
        if show: self.show()    
        return(has_game_ended)

    def show(self):
        # dispalys the board in text version
    
        for i in range(0,3):
            print (self.state[3*i:3*i+3])
        state_value = win_value if code_winner in self.state else self.state_value_function[self.state] 
        print ('Prob of winning for player:', state_value)
        sys.stdout.flush()
        print(' ')
        

class Player():
    def __init__(self, code):
        self.code = code

board = Board()

# attempts to download the Agent's parameters
try: 
    with open('TicTacToe_parameters', 'r') as file_data:
        board.state_value_function = dict(csv.reader(file_data))
        for key in board.state_value_function.keys():
            board.state_value_function[key] = float(board.state_value_function[key])
            board.td_traces[key] = 0.
        file_data.close()
    is_training = False
except IOError:
    is_training = True
    player1 = Player(code_player1)
    player2 = Player(code_player2)
    print ('Working out state-value function ...')
    sys.stdout.flush()
    for epoch in range(0,max_epochs):
        show = True if epoch % frequency_show == 0 else False
        if show: 
            print('-----------------------\n')
        while not board.check_end_and_update_state_values(show):
            board.decide()
        board.reboot()

def screen_display(Human, AI):   
    # displays the current board via Pygame 

    screen.fill(WHITE)
    pygame.draw.line(screen,(0,0,0),(100,300),(100,0))
    pygame.draw.line(screen,(0,0,0),(200,300),(200,0))
    pygame.draw.line(screen,(0,0,0),(300,100),(0,100))
    pygame.draw.line(screen,(0,0,0),(300,200),(0,200))
    for i in range(0,len(board.state)):
        if board.state[i] == AI.code:
            pygame.draw.rect(screen,BLUE,( int(i/3)*100, (i%3)*100, 100, 100))
        if board.state[i] == Human.code:
            pygame.draw.rect(screen,RED,( int(i/3)*100, (i%3)*100, 100, 100))
        if board.state[i] == code_winner:
            pygame.draw.rect(screen,YELLOW,( int(i/3)*100, (i%3)*100, 100, 100))
    pygame.display.flip()


BLACK = (  0,   0,   0)
WHITE = (255, 255, 255)
RED   = (255, 192, 203)
BLUE  = (176, 196, 222)
YELLOW= (218, 165,  32) 

# initializes Pygame
pygame.init()
pygame.display.set_caption(game_title)
screen = pygame.display.set_mode((300,300))

# launches the menu and receives instructions over which player starts first
HUMAN_STARTS = menu.launch(screen)

# the game is launched
running = True
while running: 
        
    # AI makes the first move if it starts first
    if not HUMAN_STARTS:
        AI = Player(code_player1)
        Human = Player(code_player2)
        board.decide(greedy = True)
    else:
        AI = Player(code_player2)
        Human = Player(code_player1)
        
    # updates the screen
    screen_display(Human, AI)
      
    # main game cycle
    game_on = True
    while game_on:
        
        # checks if a quit command is given 
        for event in pygame.event.get():
            if event.type == pygame.QUIT: 
                running = game_on = False
                continue
                        
            # receives the Human's decision
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_position = event.pos
                decision = int(mouse_position[0]/100) * 3 + int(mouse_position[1]/100)
                if list(board.state)[decision] == '-':
                    temp = list(board.state); temp[decision] = Human.code; board.state = ''.join(temp);
                    board.game.append(board.state)
                
                    # checks if there's a winner
                    if board.check_end_and_update_state_values(): 
                        game_on = False
                    
                    # AI makes a decision
                    if game_on:
                        board.decide(greedy = True)
                        if board.check_end_and_update_state_values(): 
                            game_on = False
            
        # display scren
        screen_display(Human, AI)                
    
    screen_display(Human, AI)
    time.sleep(1)
    board.reboot()    

# closes pygame
pygame.display.quit()
pygame.quit()

# saves the Agent's parameters
if is_training:
    with open('TicTacToe_parameters', 'w') as file_data:
        writer = csv.writer(file_data)
        for key, value in board.state_value_function.items():
            writer.writerow([key, value])
        file_data.close()  


#####################################################################################
print("The process took : ", round(time.time()-start,2), ' seconds')




    
