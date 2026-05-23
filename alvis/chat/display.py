import os
import textwrap
from config import DISPLAY_TURNS


def display_chat(history, user_input=None, response=None):
    os.system("cls" if os.name == "nt" else "clear")
    try:
        width = os.get_terminal_size().columns
    except OSError:
        width = 80
    indent = " " * 8

    def fmt(prefix, text):
        return textwrap.fill(text, width=width, initial_indent=prefix, subsequent_indent=indent)

    print("─" * min(40, width))
    print("ALVIS  |  /help · reset · quit")
    print("─" * min(40, width))
    visible = history[-(DISPLAY_TURNS * 2):]
    i = 0
    while i < len(visible) - 1:
        if visible[i]["role"] == "user" and visible[i + 1]["role"] == "assistant":
            print(fmt("Vous  : ", visible[i]["content"]))
            print(fmt("ALVIS : ", visible[i + 1]["content"]) + "\n")
            i += 2
        else:
            i += 1
    if user_input is not None:
        print(fmt("Vous  : ", user_input))
    if response is not None:
        print(fmt("ALVIS : ", response) + "\n")
