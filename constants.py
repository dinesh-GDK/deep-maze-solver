from enum import Enum

MAZE_MIN_SIZE = 5
MAZE_MAX_SIZE = 127
MODEL_INPUT_SIZE = 128


class IteratorMode(Enum):
    TRAIN = "TRAIN"
    VALIDATE = "VALIDATE"
    TEST = "TEST"
