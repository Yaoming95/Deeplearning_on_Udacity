from __future__ import print_function

from textassignment.A6_0_const import *


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - first_letter + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(dictid):
    if dictid > 0:
        return chr(dictid + first_letter - 1)
    else:
        return ' '

if __name__ == '__main__':
    print(char2id('a'), char2id('z'), char2id(' '), char2id('ï'))
    print(id2char(1), id2char(26), id2char(0))

