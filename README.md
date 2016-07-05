## TicTacToe
I applied the TD(lambda) learning method to teach an agent how to play at TicTacToe (TTT).  
The agent learns by playing against itself.   

As TTT is a deterministic and memoryless game, the agent only needs to work out a state-value function rather than a state/action-value function. In first approximation, the maximum number of states is capped at 6046, hence the number of possible states is small enough to be stored and valued.  

I applied the following updating formula:
>Q(s) <- Q(s) + alpha * [ Reward + gamma*Q(S_t+1) - Q(S_t)] * Tr(s)

### Quick Start
Download 'tictactoe v3.0.py', 'menu.py', and 'TicTacToe_parameters' to the same folder. Then, launch 'tictactoe v3.0.py'.  
You will be able to play against the agent. You can choose whether you want to start first or not. 
The agent learnt a perfect strategy already. If the 'TicTacToe_parameters' file is not in the folder, the agent will learn a strategy from scratch (it takes approx 5min).

### Requirements
* PyGame. I used version 1.9.2a0.  
You can download it from [here][1].  

<img src="Animation.gif" width="30%" />

### Resources
Here is some recommended material:
* `Reinforcement Learning: An Introduction`, PDF notes by Sutton and Barto ([link][2]).  
* `Reinforcement Learning`, online cource by Udacity ([link][3]).

[1]: http://www.pygame.org/download.shtml
[2]: http://people.inf.elte.hu/lorincz/Files/RL_2006/SuttonBook.pdf
[3]: https://www.udacity.com/course/reinforcement-learning--ud600

