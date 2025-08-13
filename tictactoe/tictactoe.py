"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return O if x_count > o_count else X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possible_actions = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                possible_actions.add((i, j))
    return possible_actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    i, j = action
    if i not in range(3) or j not in range(3) or board[i][j] != EMPTY:
        raise Exception("Invalid action")
    
    new_board = copy.deepcopy(board)
    new_board[i][j] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for row in board:
        if row == [X, X, X]:
            return X
        if row == [O, O, O]:
            return O
    
    # Check columns
    for j in range(3):
        if [board[i][j] for i in range(3)] == [X, X, X]:
            return X
        if [board[i][j] for i in range(3)] == [O, O, O]:
            return O
    
    # Check diagonals
    if [board[i][i] for i in range(3)] == [X, X, X]:
        return X
    if [board[i][i] for i in range(3)] == [O, O, O]:
        return O
    if [board[0][2], board[1][1], board[2][0]] == [X, X, X]:
        return X
    if [board[0][2], board[1][1], board[2][0]] == [O, O, O]:
        return O
    
    return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    if all(cell != EMPTY for row in board for cell in row):
        return True
    return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    winner_result = winner(board)
    if winner_result == X:
        return 1
    if winner_result == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    
    current_player = player(board)
    
    if current_player == X:
        value, move = max_value(board)
    else:
        value, move = min_value(board)
    
    return move


def max_value(board):
    """
    Returns the maximum utility and best move for X.
    """
    if terminal(board):
        return utility(board), None
    
    v = -math.inf
    best_move = None
    
    for action in actions(board):
        min_val, _ = min_value(result(board, action))
        if min_val > v:
            v = min_val
            best_move = action
    
    return v, best_move


def min_value(board):
    """
    Returns the minimum utility and best move for O.
    """
    if terminal(board):
        return utility(board), None
   
    v = math.inf
    best_move = None
 
    for action in actions(board):
        max_val, _ = max_value(result(board, action))
        if max_val < v:
            v = max_val
            best_move = action
 
    return v, best_move
