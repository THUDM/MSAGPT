def move_cursor_up(n):
    # ANSI escape code to move cursor up by n lines
    print(f"\033[{n}A", end='')

def move_cursor_down(n):
    # ANSI escape code to move cursor down by n lines
    print(f"\033[{n}B", end='')