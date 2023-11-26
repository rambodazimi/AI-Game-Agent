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
      
      
@register_agent("student_agent3")
class StudentAgent3(Agent):
    """
    A dummy class for your implementation. Feel free to use this class to
    add any helper functionalities needed for your agent.
    """

    def __init__(self):
        super(StudentAgent3, self).__init__()
        self.name = "StudentAgent3"
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }
        
    @staticmethod
    def find_allowed_dirs(chess_board, my_pos, adv_pos):
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
    def find_allowed_moves(chess_board, my_pos, adv_pos, max_step, allowed_move_list):
        """
        This method uses recursion to examine all the possible moves that student agent can go and returns the list of all possible moves
        """
        # Count the current position as an allowed move
        if my_pos not in allowed_move_list:
            allowed_move_list.append(my_pos)
        
        # Base case: (no more steps to take)
        if(max_step == 0):
            return allowed_move_list
        
        # Recursive case: (still have steps to take)
        allowed_dirs, _ = StudentAgent3.find_allowed_dirs(chess_board, my_pos, adv_pos)
        
        r, c = my_pos
        for d in allowed_dirs:
            m_r, m_c = moves[d]
            new_pos = (r + m_r, c + m_c)
            allowed_move_list = StudentAgent3.find_allowed_moves(chess_board, new_pos, adv_pos, max_step-1, allowed_move_list)
        
        return allowed_move_list
    
    @staticmethod
    def manhattan_distance(my_pos, adv_pos):
        row_A, col_A = my_pos
        row_B, col_B = adv_pos

        return abs(row_A - row_B) + abs(col_A - col_B)
    
    @staticmethod  
    def adv_in_reach(chess_board, adv_pos, allowed_moves):
        """
        Check if the opponent's square has at least one reachable side from us, if so, it returns the move we need to make.
        """
        
        # Check if any of the possible moves can reach the adversary
        for move in allowed_moves:
            for d in range(0, 4):
                if adv_pos == (move[0] + moves[d][0], move[1] + moves[d][1]) and not chess_board[move[0], move[1], d]:
                    return True, move, d
                
        return False, None, None

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
                distance_A = StudentAgent3.manhattan_distance(top_left_corner, my_pos)
            else:
                # top right
                distance_A = StudentAgent3.manhattan_distance(top_right_corner, my_pos)
        else:
            if(y_A < mid):
                # bottom left
                distance_A = StudentAgent3.manhattan_distance(bottom_left_corner, my_pos)
            else:
                # bottom right
                distance_A = StudentAgent3.manhattan_distance(bottom_right_corner, my_pos)

        # random agent
        if(x_B < mid):
            if(y_B < mid):
                # top left
                distance_B = StudentAgent3.manhattan_distance(top_left_corner, adv_pos)
            else:
                # top right
                distance_B = StudentAgent3.manhattan_distance(top_right_corner, adv_pos)
        else:
            if(y_B < mid):
                # bottom left
                distance_B = StudentAgent3.manhattan_distance(bottom_left_corner, adv_pos)
            else:
                # bottom right
                distance_B = StudentAgent3.manhattan_distance(bottom_right_corner, adv_pos)

        return distance_A, distance_B

    @staticmethod
    def min_max_normalize(value, min_val, max_val):
        """
        This method applies the min max normalization to normalize the heuristic scores
        """
        normalized_value = (value - min_val) / (max_val - min_val)
        return normalized_value

    @staticmethod
    def calculate_score(chess_board, pos, adv_pos, max_step):
        """
        This method is used for calculating the total score at each turn based on different heuristics:
        1. Number of allowed directions (Maximize)
        At each turn, calculate the number of possible directions that student agent can choose to move (0 to 4)

        2. Total possible moves (Maximize)
        At each turn, calculate the total number of possible moves the student agent can move (approximately from 1 to M^2)

        3. Distance from the nearest corner of the student agent (Maximize)

        Finally, we apply min max normalization technique and choose weights to prioritize some heuristics and return the final score
        (For example, number of allowed directions seems to be more important than the distance from the opponent; thus, we consider a higher weight for that) 
        """
        _, heuristic_1 = StudentAgent3.find_allowed_dirs(chess_board, pos, adv_pos)
        
        heuristic_2 = len(StudentAgent3.find_allowed_moves(chess_board, pos, adv_pos, max_step, []))

        heuristic_3,_ = StudentAgent3.distance_from_nearest_corner(chess_board, pos, adv_pos)

        # calculating the total score #####
        normalized_heuristic_1 = StudentAgent3.min_max_normalize(heuristic_1, 0, 4)
        normalized_heuristic_2 = StudentAgent3.min_max_normalize(heuristic_2, 0, chess_board.shape[0] ** 2)
        normalized_heuristic_3 = StudentAgent3.min_max_normalize(heuristic_3, 0, chess_board.shape[0])
        weights = [0.5, 0.3, 0.2]
        total_score = normalized_heuristic_1 * weights[0] + normalized_heuristic_2 * weights[1] + normalized_heuristic_3 * weights[2]

        return total_score * 100
    
    @staticmethod
    def find_best_dir(chess_board, my_pos, adv_pos, allowed_dirs, max_step):
        
        # Default to the first allowed direction
        best_dir = allowed_dirs[0]

        # If there is only one allowed direction, return it
        if len(allowed_dirs) == 1:
            return best_dir

        max_score = -1000
        for dir in allowed_dirs:
            chess_board_tmp = deepcopy(chess_board)
            chess_board_tmp[my_pos[0], my_pos[1], dir] = True # Put a wall in the direction we are considering
            our_moves = StudentAgent3.find_allowed_moves(chess_board_tmp, my_pos, adv_pos, max_step, [])
            opponent_moves = StudentAgent3.find_allowed_moves(chess_board_tmp, adv_pos, my_pos, max_step, [])
            score = len(our_moves) - len(opponent_moves)
            if score > max_score:
                max_score = score
                best_dir = dir
                
        return best_dir
        
    

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
        
        # Find all the possible moves we can make
        allowed_moves = StudentAgent3.find_allowed_moves(chess_board, my_pos, adv_pos, max_step, [])
        
        # Before anything, check if we can trap the opponent in a 1X1 square, and if so, do it.
        adv_in_reach, move, dir = StudentAgent3.adv_in_reach(chess_board, adv_pos, allowed_moves)
        
        if adv_in_reach:
            adv_open_dirs = []
            for d in range(0, 4):
                if chess_board[adv_pos[0], adv_pos[1], d] == False:
                    adv_open_dirs.append(d)
    
            if len(adv_open_dirs) == 1:

                time_taken = time.time() - start_time
                print("My AI's turn took ", time_taken, "seconds.")
                return move, dir

        # Find the best move to make using our heuristic score function
        best_move = my_pos
        runner_up_move = my_pos
        best_move_score = -1000
        runner_up_score = -1000
        for move in allowed_moves:
            score = StudentAgent3.calculate_score(chess_board, move, adv_pos, max_step)
            if score > best_move_score:
                _, allowed_dirs_nb = StudentAgent3.find_allowed_dirs(chess_board, move, adv_pos)
                
                # Avoid trapping ourselves or putting ourselves in a corner, unless we have no other choice
                if allowed_dirs_nb >= 3:
                    best_move_score = score
                    best_move = move
                    
                elif allowed_dirs_nb == 2 and score > runner_up_score:
                    runner_up_score = score
                    runner_up_move = move
                    
        if best_move_score == -1000:  # If no moves prevent us from being easily trapped next turn, we take the runner up best move
            best_move = runner_up_move

        # Find the best direction to put a wall
        allowed_dirs, _ = StudentAgent3.find_allowed_dirs(chess_board, best_move, adv_pos)
        
        best_dir = StudentAgent3.find_best_dir(chess_board, best_move, adv_pos, allowed_dirs, max_step)

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")
        return best_move, best_dir
