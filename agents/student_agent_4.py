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


@register_agent("student_agent4")
class StudentAgent4(Agent):
    """
    Our final student_agent implementation.
    """

    def __init__(self):
        super(StudentAgent4, self).__init__()
        self.name = "StudentAgent4"
        self.autoplay = True
        self.dir_map = {
            "u": 0,
            "r": 1,
            "d": 2,
            "l": 3,
        }

    @staticmethod
    def find_allowed_dirs(chess_board, my_pos, adv_pos):
        """
        This method calculates the number of possible directions that the student
        agent can go and returns both the value and possible dirs as a list.

        returns: A tuple of
        allowed_dirs: The list of possible directions that the student agent can go
        len(allowed_dirs): The number of possible directions that the student agent can go

        parameters:
        chess_board: The current state of the game board
        my_pos: The current position of the student agent
        adv_pos: The current position of the opponent
        """
        row, col = my_pos
        # Build a list of the moves we can make
        allowed_dirs = [
            d
            for d in range(0, 4)  # 4 moves possible (up, down, left, right)
            if not chess_board[row, col, d]
            and not adv_pos  # chess_board True means wall
            == (row + moves[d][0], col + moves[d][1])
        ]  # cannot move through Adversary
        return allowed_dirs, len(allowed_dirs)

    @staticmethod
    def find_allowed_moves(chess_board, my_pos, adv_pos, max_step, allowed_move_list):
        """
        This method uses recursion to examine all the possible moves that student agent
        can do and returns the list of all possible moves.

        returns:
        allowed_move_list: The list of all the possible moves that the student agent can make

        parameters:
        chess_board: The current state of the game board
        my_pos: The current position of the student agent
        adv_pos: The current position of the opponent
        max_step: The maximum number of steps that can be taken in the game
        allowed_move_list: The current list of all the possible moves that the student agent can make
        """
        # Count the current position as an allowed move
        if my_pos not in allowed_move_list:
            allowed_move_list.append(my_pos)

        # Base case: (no more steps to take)
        if max_step == 0:
            return allowed_move_list

        # Recursive case: (still have steps to take)
        allowed_dirs, _ = StudentAgent4.find_allowed_dirs(chess_board, my_pos, adv_pos)

        r, c = my_pos
        for d in allowed_dirs:
            m_r, m_c = moves[d]
            new_pos = (r + m_r, c + m_c)
            allowed_move_list = StudentAgent4.find_allowed_moves(
                chess_board, new_pos, adv_pos, max_step - 1, allowed_move_list
            )

        return allowed_move_list

    @staticmethod
    def manhattan_distance(my_pos, adv_pos):
        """
        This returns the manhanntan distance between two positions.

        returns:
        distance: The manhattan distance between the two positions

        parameters:
        my_pos: The position of the student agent
        adv_pos: The position of the opponent
        """
        row_A, col_A = my_pos
        row_B, col_B = adv_pos

        return abs(row_A - row_B) + abs(col_A - col_B)

    @staticmethod
    def adv_in_reach(chess_board, adv_pos, allowed_moves):
        """
        Check if the opponent's square has at least one reachable side from us, if so,
        it returns the move and direction we need to make to reach it.

        returns: A tuple of
        True if the opponent is in reach/False if the opponent is not in reach,
        move: The position that the student agent needs to make to reach the opponent,
        dir: The direction that the student agent needs to go to reach the opponent from the above move

        parameters:
        chess_board: The current state of the game board
        adv_pos: The current position of the opponent
        allowed_moves: The list of all the possible moves that the student agent can make
        """

        # Check if any of the possible moves can reach the adversary
        for move in allowed_moves:
            for d in range(0, 4):
                if (
                    adv_pos == (move[0] + moves[d][0], move[1] + moves[d][1])
                    and not chess_board[move[0], move[1], d]
                ):
                    return True, move, d

        return False, None, None

    @staticmethod
    def distance_from_nearest_corner(chess_board, pos):
        """
        This method calculates the distance from the closest corner for the student agent.

        returns:
        distance_A: The distance from the nearest corner for the student agent

        parameters:
        chess_board: The current state of the game board
        my_pos: The current position of the student agent
        """
        length = chess_board.shape[0]
        top_left_corner = (0, 0)
        bottom_left_corner = (chess_board.shape[0] - 1, 0)
        top_right_corner = (0, chess_board.shape[1] - 1)
        bottom_right_corner = (chess_board.shape[0] - 1, chess_board.shape[1] - 1)

        x_A, y_A = pos

        mid = int(length / 2)

        # student agent
        if x_A < mid:
            if y_A < mid:
                # top left
                distance_A = StudentAgent4.manhattan_distance(top_left_corner, pos)
            else:
                # top right
                distance_A = StudentAgent4.manhattan_distance(top_right_corner, pos)
        else:
            if y_A < mid:
                # bottom left
                distance_A = StudentAgent4.manhattan_distance(bottom_left_corner, pos)
            else:
                # bottom right
                distance_A = StudentAgent4.manhattan_distance(bottom_right_corner, pos)

        return distance_A

    @staticmethod
    def min_max_normalize(value, min_val, max_val):
        """
        This method applies min max normalization to normalize the heuristic scores.

        returns:
        normalized_value: The normalized value of the heuristic

        parameters:
        value: The value to be normalized
        min_val: The minimum value of the range
        max_val: The maximum value of the range
        """
        normalized_value = (value - min_val) / (max_val - min_val)
        return normalized_value

    @staticmethod
    def calculate_pos_score(chess_board, pos, adv_pos, max_step):
        """
        This method is used for calculating the total score of a position move
        at each turn based on different heuristics:

        1. Number of allowed directions (Maximize)
        At each turn, calculate the number of possible directions
        that student agent can choose to move (0 to 4).

        2. Total possible moves (Maximize)
        At each turn, calculate the total number of possible moves the student
        agent can move (approximately from 1 to M^2)

        3. Distance from the nearest corner of the student agent (Maximize)
        Finally, we apply min max normalization technique and choose weights
        to prioritize some heuristics and return the final score

        returns:
        total_score: The total score of the position move

        Parameters:
        chess_board: The current state of the game board
        pos: The current position of the student agent
        adv_pos: The current position of the opponent
        max_step: The maximum number of steps that can be taken in the game
        """

        _, heuristic_1 = StudentAgent4.find_allowed_dirs(chess_board, pos, adv_pos)

        heuristic_2 = len(
            StudentAgent4.find_allowed_moves(chess_board, pos, adv_pos, max_step, [])
        )

        heuristic_3 = StudentAgent4.distance_from_nearest_corner(chess_board, pos)

        # calculating the total score #####
        normalized_heuristic_1 = StudentAgent4.min_max_normalize(heuristic_1, 0, 4)
        normalized_heuristic_2 = StudentAgent4.min_max_normalize(
            heuristic_2, 0, chess_board.shape[0] ** 2
        )
        normalized_heuristic_3 = StudentAgent4.min_max_normalize(
            heuristic_3, 0, chess_board.shape[0]
        )
        weights = [0.5, 0.3, 0.2]
        total_score = (
            normalized_heuristic_1 * weights[0]
            + normalized_heuristic_2 * weights[1]
            + normalized_heuristic_3 * weights[2]
        )

        return total_score * 100

    @staticmethod
    def calculate_dir_score(chess_board, pos, adv_pos, max_step, dir):
        """
        This method is used for calculating the total score of a direction move
        at each turn based on one heuristic:

        1. Difference of allowed moves between us and the opponent. (Maximize)
        This heuristic finds the direction that gives us the best edge over the opponent in terms of allowed moves.

        returns:
        score: The total score of the direction move

        Parameters:
        chess_board: The current state of the game board
        pos: The position from which we aim to put a barrier
        adv_pos: The current position of the opponent
        max_step: The maximum number of steps that can be taken in the game
        dir: The direction that we are considering to put a wall
        """

        chess_board_tmp = deepcopy(chess_board)
        chess_board_tmp[
            pos[0], pos[1], dir
        ] = True  # Put a wall in the direction we are considering

        our_moves = StudentAgent4.find_allowed_moves(
            chess_board_tmp, pos, adv_pos, max_step, []
        )
        opponent_moves = StudentAgent4.find_allowed_moves(
            chess_board_tmp, adv_pos, pos, max_step, []
        )
        
        nb_extended_walls = 0
        if dir == 0: #up
            if pos[0] - 1 < chess_board.shape[0] and chess_board[pos[0] - 1, pos[1], 1]  : #if there is a wall on the right of the top position
                nb_extended_walls += 1
                
            if  pos[0] - 1 < chess_board.shape[0] and chess_board[pos[0] - 1, pos[1], 3]: #if there is a wall on the left of the top position
                nb_extended_walls += 1
                
            if  pos[1] - 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] - 1, 0]: #if there is a wall on the top of the left position
                nb_extended_walls += 1
                
            if pos[1] + 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] + 1, 0]  : #if there is a wall on the top of the right position
                nb_extended_walls += 1
                
            if chess_board[pos[0], pos[1], 1] :  #if there is a wall on the right of the current position
                nb_extended_walls += 1
                
            if chess_board[pos[0], pos[1], 3] : #if there is a wall on the left of the current position
                nb_extended_walls += 1
            
        if dir == 1: #right
            if chess_board[pos[0], pos[1], 0] : #if there is a wall on the top of the current position
                nb_extended_walls += 1
                
            if chess_board[pos[0], pos[1], 2] : #if there is a wall on the bottom of the current position
                nb_extended_walls += 1
                
            if pos[0] - 1 < chess_board.shape[0] and chess_board[pos[0] - 1, pos[1], 1] : #if there is a wall on the right of the top position
                nb_extended_walls += 1
                
            if  pos[0] + 1 < chess_board.shape[0] and chess_board[pos[0] + 1, pos[1], 1] : #if there is a wall on the right of the bottom position
                nb_extended_walls += 1
                
            if  pos[1] + 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] + 1, 0]: #if there is a wall on the top of the right position
                nb_extended_walls += 1
                
            if  pos[1] + 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] + 1, 2]: #if there is a wall on the bottom of the right position
                nb_extended_walls += 1

        if dir == 2: #down
            if chess_board[pos[0], pos[1], 1]  :  #if there is a wall on the right of the current position
                nb_extended_walls += 1
                
            if chess_board[pos[0], pos[1], 3] : #if there is a wall on the left of the current position
                nb_extended_walls += 1
                
            if pos[1] + 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] + 1, 2] : #if there is a wall on the bottom of the right position
                nb_extended_walls += 1
                
            if pos[1] - 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] - 1, 2] : #if there is a wall on the bottom of the left position
                nb_extended_walls += 1
                
            if pos[0] + 1 < chess_board.shape[0] and chess_board[pos[0] + 1, pos[1], 1] : #if there is a wall on the right of the bottom position
                nb_extended_walls += 1
                
            if pos[0] + 1 < chess_board.shape[0] and chess_board[pos[0] + 1, pos[1], 3] : #if there is a wall on the left of the bottom position
                nb_extended_walls += 1
                
        if dir == 3: #left
            if chess_board[pos[0], pos[1], 0] : #if there is a wall on the top of the current position
                nb_extended_walls += 1
                
            if chess_board[pos[0], pos[1], 2] : #if there is a wall on the bottom of the current position
                nb_extended_walls += 1
                
            if pos[0] - 1 < chess_board.shape[0] and chess_board[pos[0] - 1, pos[1], 3] : #if there is a wall on the left of the top position
                nb_extended_walls += 1
                
            if pos[0] + 1 < chess_board.shape[0] and chess_board[pos[0] + 1, pos[1], 3] : #if there is a wall on the left of the bottom position
                nb_extended_walls += 1
                
            if pos[1] - 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] - 1, 0] : #if there is a wall on the top of the left position
                nb_extended_walls += 1
                
            if pos[1] - 1 < chess_board.shape[0] and chess_board[pos[0], pos[1] - 1, 2] : #if there is a wall on the bottom of the left position
                nb_extended_walls += 1
        
        heur1 = len(our_moves) - len(opponent_moves)
        heur2 = nb_extended_walls
        weights = [0.5, 0.5]


        return StudentAgent4.min_max_normalize(heur1, 0, chess_board.shape[0] ** 2) * weights[0] + StudentAgent4.min_max_normalize(heur2, 0, 6) * weights[1]

    def step(self, chess_board, my_pos, adv_pos, max_step):
        """
        This method is called every turn by the game engine and returns the move
        and direction that the student agent wants to make.

        It first checks if the opponent is in reach and easily trappable, if so,
        it returns the move and direction to trap the opponent in a 1X1 square.

        Otherwise, it works as a 2-step heuristic search.
        Firstly, it searches for the best move using a heuristic function "calculate_pos_score".
        Secondly, it searches the best direction to put a wall using another heuristic function "calculate_dir_score".

        returns: A tuple of
        best_move: The best move that the student agent wants to make
        best_dir: The best direction that the student agent wants to put a wall

        Parameters:
        chess_board: The current state of the game board
        my_pos: The current position of the student agent
        adv_pos: The current position of the opponent
        max_step: The maximum number of steps that can be taken in the game
        """
        start_time = time.time()

        # Find all the possible moves we can make
        allowed_moves = StudentAgent4.find_allowed_moves(
            chess_board, my_pos, adv_pos, max_step, []
        )

        # Check if the opponent is in reach
        adv_in_reach, from_pos, from_dir = StudentAgent4.adv_in_reach(
            chess_board, adv_pos, allowed_moves
        )

        # Check if the opponent is cornered (3 barriers around it)
        if adv_in_reach:
            adv_open_dirs = []
            for d in range(0, 4):
                if chess_board[adv_pos[0], adv_pos[1], d] == False:
                    adv_open_dirs.append(d)

            if (
                len(adv_open_dirs) == 1
            ):  # Return winning move right away if the opponent is cornered
                time_taken = time.time() - start_time
                print("My AI's turn took ", time_taken, "seconds.")
                return from_pos, from_dir

        # Find the best move to make using our heuristic score function
        best_move = my_pos
        runner_up_move = my_pos
        best_move_score = -1000
        runner_up_score = -1000
        for move in allowed_moves:
            score = StudentAgent4.calculate_pos_score(
                chess_board, move, adv_pos, max_step
            )
            if score > best_move_score:
                _, allowed_dirs_nb = StudentAgent4.find_allowed_dirs(
                    chess_board, move, adv_pos
                )

                # Avoid trapping ourselves or putting ourselves in a corner, unless we have no other choice
                if allowed_dirs_nb >= 3:
                    best_move_score = score
                    best_move = move

                elif allowed_dirs_nb == 2 and score > runner_up_score:
                    runner_up_score = score
                    runner_up_move = move

        if (
            best_move_score == -1000
        ):  # If no moves prevent us from being easily trapped next turn, we take the runner up best move
            best_move = runner_up_move

        # Find the best direction to put a barrier
        allowed_dirs, _ = StudentAgent4.find_allowed_dirs(
            chess_board, best_move, adv_pos
        )

        # Default to the first allowed direction
        best_dir = allowed_dirs[0]

        if (
            len(allowed_dirs) > 1
        ):  # Avoid checking scores if there is only one allowed direction
            max_score = -1000
            for dir in allowed_dirs:
                score = StudentAgent4.calculate_dir_score(
                    chess_board, best_move, adv_pos, max_step, dir
                )
                if score > max_score:
                    max_score = score
                    best_dir = dir

        time_taken = time.time() - start_time
        print("My AI's turn took ", time_taken, "seconds.")
        return best_move, best_dir
