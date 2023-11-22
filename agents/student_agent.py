# Student agent

# Authors:
# Rambod Azimi
# Matthew Spagnuolo
 
from agents.agent import Agent
from store import register_agent
import sys
import numpy as np
from copy import deepcopy
import time


# Moves (Up, Right, Down, Left)
moves = ((-1, 0), (0, 1), (1, 0), (0, -1))
      
      
@register_agent("student_agent")
class StudentAgent(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent, self).__init__()
        self.name = "StudentAgent"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        
    @staticmethod
    def number_of_allowed_dirs(chess_board, my_pos, adv_pos):
        """
        This method calculates the number of possible directions that the student agent can go and returns both the value and possible dirs as a list
        """
        row, col = my_pos
        # Build a list of the moves we can make
        allowed_dirs = [ d                                
            for d in range(0, 4) # 4 moves possible (up, down, left, right)
            if not chess_board[row, col, d] and # chess_board True means wall
            not adv_pos == (row + moves[d][0], col + moves[d][1])] # cannot move through Adversary
        return allowed_dirs, len(allowed_dirs)
    
    @staticmethod
    def total_allowed_moves(chess_board, my_pos, adv_pos, max_step, allowed_dirs):
        """
        This method uses recursion to examine all the possible moves that student agent can go and returns the value
        """
        if(max_step == 0):
            return 0
        
        allowed_dirs, _ = StudentAgent.number_of_allowed_dirs(chess_board, my_pos, adv_pos)

        if(len(allowed_dirs) == 0):
            return 0
        elif(len(allowed_dirs) == 1):
            return 1 + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[0]], adv_pos, max_step-1, allowed_dirs)
        elif(len(allowed_dirs) == 2):
            return 1 + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[0]], adv_pos, max_step-1, allowed_dirs) + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[1]], adv_pos, max_step-1, allowed_dirs)
        elif(len(allowed_dirs) == 3):
            return 1 + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[0]], adv_pos, max_step-1, allowed_dirs) + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[1]], adv_pos, max_step-1, allowed_dirs) + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[2]], adv_pos, max_step-1, allowed_dirs)
        elif(len(allowed_dirs) == 4):
            return 1 + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[0]], adv_pos, max_step-1, allowed_dirs) + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[1]], adv_pos, max_step-1, allowed_dirs) + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[2]], adv_pos, max_step-1, allowed_dirs) + StudentAgent.total_allowed_moves(chess_board, moves[allowed_dirs[3]], adv_pos, max_step-1, allowed_dirs)
        else:
            return 0
    
    @staticmethod
    def manhattan_distance(my_pos, adv_pos):
        row_A, col_A = my_pos
        row_B, col_B = adv_pos

        return abs(row_A - row_B) + abs(col_A - col_B)
    
    @staticmethod  
    def adv_in_reach(chess_board, my_pos, adv_pos, max_step, last_pos):
        """
        Check if the opponent's square has at least one reachable side from us, recursively.
        """
        r, c = my_pos
        
        #Base cases:
        if r == adv_pos[0] and c == adv_pos[1]:
            return True
        
        if max_step == 0:
            return False
        
        #Recursive case:
        
        # Build a list of the moves we can make
        allowed_dirs = [ d                                
            for d in range(0, 4) # 4 moves possible (up, down, left, right)
            if not chess_board[r, c, d]] # chess_board True means wall
            
        # Check all possible moves for a reachable path to the adversary
        in_reach = False
        for d in allowed_dirs:
            m_r, m_c = moves[d]
            new_pos = (r + m_r, c + m_c)
            if new_pos != last_pos:
                in_reach = in_reach or StudentAgent.adv_in_reach(chess_board, new_pos, adv_pos, max_step-1, my_pos)
            
        return in_reach

    @staticmethod
    def distance_from_nearest_corner(chess_board, my_pos, adv_pos):
        """
        This method calculates the distance from the closest corner for both the player (student agent) and opponent and returns a tuple
        """
        length = chess_board.shape[0]
        top_left_corner = (0, 0)
        bottom_left_corner = (chess_board.shape[0] - 1, 0)
        top_right_corner = (0, chess_board.shape[1] - 1)
        bottom_right_corner = (chess_board.shape[0] - 1, chess_board.shape[1] - 1)

        x_A, y_A = my_pos
        x_B, y_B = adv_pos

        mid = int(length / 2)

        # student agent
        if(x_A < mid):
            if(y_A < mid):
                # top left
                distance_A = StudentAgent.manhattan_distance(top_left_corner, my_pos)
            else:
                # top right
                distance_A = StudentAgent.manhattan_distance(top_right_corner, my_pos)
        else:
            if(y_A < mid):
                # bottom left
                distance_A = StudentAgent.manhattan_distance(bottom_left_corner, my_pos)
            else:
                # bottom right
                distance_A = StudentAgent.manhattan_distance(bottom_right_corner, my_pos)

        # random agent
        if(x_B < mid):
            if(y_B < mid):
                # top left
                distance_B = StudentAgent.manhattan_distance(top_left_corner, adv_pos)
            else:
                # top right
                distance_B = StudentAgent.manhattan_distance(top_right_corner, adv_pos)
        else:
            if(y_B < mid):
                # bottom left
                distance_B = StudentAgent.manhattan_distance(bottom_left_corner, adv_pos)
            else:
                # bottom right
                distance_B = StudentAgent.manhattan_distance(bottom_right_corner, adv_pos)

        return distance_A, distance_B

    @staticmethod
    def min_max_normalize(value, min_val, max_val):
        """
        This method applies the min max normalization to normalize the heuristic scores
        """
        normalized_value = (value - min_val) / (max_val - min_val)
        return normalized_value

    @staticmethod
    def calculate_score(chess_board, my_pos, adv_pos, max_step):
        """
        This method is used for calculating the total score at each turn based on different heuristics:
        1. Number of allowed directions
        At each turn, calculate the number of possible directions that student agent can choose to move (0 to 4)

        2. Total possible moves
        At each turn, calculate the total number of possible moves the student agent can move (approximately from 1 to M^2)

        3. Manhattan Distance
        At each turn, calculate the manhattan distance from the student agent to the random agent (approximately 1 to 2*M)

        4. Distance from the nearest corner (student agent)

        5. Distance from the nearest corner (opponent agent)

        Finally, we apply min max normalization technique and choose weights to prioritize some heuristics and return the final score
        (For example, number of allowed directions seems to be more important than the distance from the opponent; thus, we consider a higher weight for that) 
        """
        _, heuristic_1 = StudentAgent.number_of_allowed_dirs(chess_board, my_pos, adv_pos)
        # print(f"Number of allowed dirs for student agent: {heuristic_1}")

        heuristic_2 = 0
        for i in range(0, max_step+1):
            heuristic_2 += StudentAgent.total_allowed_moves(chess_board, my_pos, adv_pos, i, [])
        # print(f"Total number of allowed moves for student agent: {heuristic_2}")

        heuristic_3 = StudentAgent.manhattan_distance(my_pos, adv_pos)
        # print(f"Manhattan distance between A and B: {heuristic_3}")

        heuristic_4, heuristic_5 = StudentAgent.distance_from_nearest_corner(chess_board, my_pos, adv_pos)
        # print(f"Player A distance from the nearest corner: {heuristic_4}")
        # print(f"Player B distance from the nearest corner: {heuristic_5}")

        # calculating the total score #####
        normalized_heuristic_1 = StudentAgent.min_max_normalize(heuristic_1, 0, 4)
        normalized_heuristic_2 = StudentAgent.min_max_normalize(heuristic_2, 0, chess_board.shape[0] ** 2)
        normalized_heuristic_3 = StudentAgent.min_max_normalize(heuristic_3, 1, chess_board.shape[0] * 2)
        normalized_heuristic_4 = StudentAgent.min_max_normalize(heuristic_4, 0, chess_board.shape[0])
        normalized_heuristic_5 = StudentAgent.min_max_normalize(heuristic_5, 0, chess_board.shape[0])
        weights = [5, 3, 0, 2, 1]
        total_score = normalized_heuristic_1 * weights[0] + normalized_heuristic_2 * weights[1] + normalized_heuristic_3 * weights[2] + normalized_heuristic_4 * weights[3] - normalized_heuristic_5 * weights[4]

        return total_score * 100
    

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        Implement the step function of your agent here.
        You can use the following variables to access the chess board:
        - chess_board: a numpy array of shape (x_max, y_max, 4)
        - my_pos: a tuple of (x, y)
        - adv_pos: a tuple of (x, y)
        - max_step: an integer

        You should return a tuple of ((x, y), dir),
        where (x, y) is the next position of your agent and dir is the direction of the wall
        you want to put on.

        Please check the sample implementation in agents/random_agent.py or agents/human_agent.py for more details.
        """
        start_time = time.time()

        #total_score = calculate_score(chess_board, my_pos, adv_pos, max_step)
        #print(f"TOTAL SCORE: {total_score:.2f}")
        
        
        # Before anything, check if we can trap the opponent in a 1X1 square, and if so, do it.
        if StudentAgent.adv_in_reach(chess_board, my_pos, adv_pos, max_step, my_pos):
            print("Adversary is in reach!")
            adv_open_dirs = []
            for d in range(0, 4):
                if chess_board[adv_pos[0], adv_pos[1], d] == False:
                    adv_open_dirs.append(d)
    
            if len(adv_open_dirs) == 1:
                print("Adversary is surrounded by 3 walls, we can win!") 
                # find our next move to trap the adversary
                my_pos = (adv_pos[0] + moves[adv_open_dirs[0]][0], adv_pos[1] + moves[adv_open_dirs[0]][1])
                if adv_open_dirs[0] <= 2:
                    dir = adv_open_dirs[0] + 2
                else:
                    dir = adv_open_dirs[0] - 2
                    
                time_taken = time.time() - start_time
                print("My AI's turn took ", time_taken, "seconds.")
                return my_pos, dir


        allowed_dirs, _ = StudentAgent.number_of_allowed_dirs(chess_board, my_pos, adv_pos)

        # trying every possible move, calculate the total score, choose move with the highest score
        max_score = -1000
        best_r = 0
        best_c = 0
        best_dir = 0
        for step in range(max_step + 1): # try every possible number of steps
            r, c = my_pos
            if len(allowed_dirs) == 0:
                # If no possible move, we must be enclosed by our Adversary
                break
            for dir in range(len(allowed_dirs)): # try every possible directions
                m_r, m_c = moves[allowed_dirs[dir]]
                #my_pos = (r + m_r, c + m_c)
                total_score = StudentAgent.calculate_score(chess_board, (r + m_r, c + m_c), adv_pos, step)
                if(total_score > max_score):
                    max_score = total_score
                    best_r = r + m_r
                    best_c = c + m_c
                    best_dir = allowed_dirs[dir]
                    

        my_pos = (best_r, best_c)

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")
        return my_pos, best_dir
