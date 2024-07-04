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
    xCount = 0
    oCount = 0
    for i in range(0, len(board)):
        for j in range(0, len(board[0])):
            if board[i][j] == X:
                xCount += 1
            elif board[i][j] == O:
                oCount += 1
    if xCount > oCount:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    moves = set()
    for i in range(0, len(board)):
        for j in range(0, len(board[0])):
            if board[i][j] == EMPTY:
                moves.add((i, j))
    return moves


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Raise exception if action is invalid
    if board[action[0]][action[1]] != EMPTY:
        raise Exception('Not a valid move')
    boardResult = copy.deepcopy(board)
    boardResult[action[0]][action[1]] = player(board)
    return boardResult


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Check rows
    for i in range(0, len(board)):
        if board[i][0] != EMPTY and board[i][0] == board[i][1] == board[i][2]:
            return board[i][0]
    # Check columns
    for j in range(0, len(board[0])):
        if board[0][j] != EMPTY and board[0][j] == board[1][j] == board[2][j]:
            return board[0][j]
    # Check diagonals
    if board[0][0] != EMPTY and board[0][0] == board[1][1] == board[2][2]:
        return board[0][0]
    elif board[0][2] != EMPTY and board[0][2] == board[1][1] == board[2][0]:
        return board[0][2]
    # No winner
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # If there is a winner - game over
    if winner(board):
        return True
    # If there isn't a winner but also no moves left - game over
    elif len(actions(board)) == 0:
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) == X:
            return 1
        elif winner(board) == O:
            return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None
    else:
        move = None
        v = float('-inf') if player(board) == X else float('inf')
        for action in actions(board):
            if player(board) == X:
                minValue = min_value(result(board, action))
                v = max(v, minValue)
                if v == minValue:
                    move = action
            elif player(board) == O:
                maxValue = max_value(result(board, action))
                v = min(v, maxValue)
                if v == maxValue:
                    move = action
        return move


def max_value(board, alpha=float('-inf'), beta=float('inf')):
    v = float('-inf')
    if terminal(board):
        return utility(board)
    for action in actions(board):
        minValue = min_value(result(board, action), alpha, beta)
        v = max(v, minValue)
        alpha = max(v, alpha)
        if alpha >= beta:
            return v
    return v


def min_value(board, alpha=float('-inf'), beta=float('inf')):
    v = float('inf')
    if terminal(board):
        return utility(board)
    for action in actions(board):
        maxValue = max_value(result(board, action), alpha, beta)
        v = min(v, maxValue)
        beta = min(v, beta)
        if alpha >= beta:
            return v
    return v
