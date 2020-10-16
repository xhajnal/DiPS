import textwrap


## source:  https://stackoverflow.com/questions/1166317/python-textwrap-library-how-to-preserve-line-breaks
from tkinter import messagebox


class DocumentWrapper(textwrap.TextWrapper):

    def wrap(self, text):
        """ Returns the text with lines split if longer than given size """
        split_text = text.split('\n')
        lines = [line for para in split_text for line in textwrap.TextWrapper.wrap(self, para)]
        return lines


## Callback function (but can be used also inside the GUI class)
def show_message(typee, where, message):
    if typee == 1 or str(typee).lower() == "error":
        messagebox.showerror(where, message)
    if typee == 2 or str(typee).lower() == "warning":
        messagebox.showwarning(where, message)
    if typee == 3 or str(typee).lower() == "info":
        messagebox.showinfo(where, message)
