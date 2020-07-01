#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Format output

column_width = [28, 48]
char_corner = '-'
char_vline = ' '
char_line = '-'
char_wait = '|'


def line(flush=True):
    """
    Return a separating line
    """

    str_line = char_corner + char_line * column_width[0] + char_corner
    str_line += char_line * column_width[1] + char_corner

    if flush is True:
        print(str_line)
        pass
    else:
        return str_line + '\n'


def row(left_value, right_value, flush=True):
    """
    Returns a table row with left. and right values in corresponding columns.

    :left_value (str, int, float or bool): the value to be printed in the left
    column

    :right_value (str, int, float or bool): the value to be printed in the
    left column

    """

    left_value = left_value if type(left_value) is str else str(left_value)
    right_value = right_value if type(right_value) is str else str(right_value)

    left_value = left_value[:column_width[0] - 5] + \
        (left_value[column_width[0] - 5:] and '...')
    right_value = right_value[:column_width[1] - 5] + \
        (right_value[column_width[1] - 5:] and '...')

    left_value = char_vline + ' ' + left_value
    right_value = char_vline + ' ' + right_value

    str_row = left_value.ljust(column_width[0]) + ' '
    str_row += right_value.ljust(column_width[1]) + ' ' + char_vline

    if flush is True:
        print(str_row)
        pass
    else:
        return str_row + '\n'


def full_row(value, flush=True):
    """
    Returns a table row with left. and right values in corresponding columns.

    :value (str, int, float or bool): the value to be printed in the
    full-column

    """

    full_width = column_width[0] + column_width[1] - 1
    value = value if type(value) is str else str(value)

    value = value[:full_width - 3] + (value[full_width - 3:] and '...')
    str_row = char_vline + ' ' + value.ljust(full_width) + ' ' + char_vline

    if flush is True:
        print(str_row)
        pass
    else:
        return str_row + '\n'


class waitbar():
    """
    Add a equal sign ('=') to the current line in order to end the line with
    a right column sign.
    """

    def __init__(self, title, max):
        """
        Returns a table row with progress bar title in left column, and
        initialized waitbar right column.

        :left_value (str, int, float or bool): the value to be printed in the
        left column

        """

        self.max = max
        self.first = True

        title = title if type(title) is str else str(title)
        title = title[:column_width[0] - 5] + \
            (title[column_width[0] - 5:] and '...')

        title = char_vline + ' ' + title

        str_row = title.ljust(column_width[0]) + ' ' + char_vline + ' '

        print(str_row, end='', flush=True)

    def progress(self, value):

        add = int((value + 1) / self.max * (column_width[1] - 2))

        if self.first is True:
            self.step = add
            print(char_wait * add, end='', flush=True)
            self.first = False
        else:
            print(char_wait * (add - self.step), end='', flush=True)
            self.step = add

        if self.step == column_width[1] - 2:
            print(' ' + char_vline, flush=True)


if __name__ == '__main__':

    line()
    row('Voiture', '%' * 46)
    row('Voiture', '%' * 46)
    line()
    full_row('Voiture' * 20)
    line()
    waitbar = waitbar('Progress', 21346)
    for i in range(21346):
        waitbar.progress(i)
    line()
