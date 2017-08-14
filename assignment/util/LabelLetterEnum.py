from enum import Enum

class LabelLetterEnum(Enum):
    A = 0
    B = 1
    C = 2
    D = 3
    E = 4
    F = 5
    G = 6
    H = 7
    I = 8
    J = 9

    def get_letter(self):
        return self.name
