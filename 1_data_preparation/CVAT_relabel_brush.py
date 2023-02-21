import queue
import threading
import tkinter as tk
import pyautogui
import time


"""This program is designed to allow users to generate a set of keyboard shortcuts with a graphical user interface. 
The program leverages the reach of the pyautogui library, and provides a visual representation of each hotkey using 
the Tkinter module.

The program begins by constructing the initial window for the program. A queue is then created to hold processed 
values. This will be used as an identifier when passing keypresses from the Tkinter interface to pyautogui.

A loop is then required to monitor the state of the queue, handle any changes, and send the correct key combination 
to pyautogui. When beginning, the system has no value set in the queue and so this ‘1’ value is sent instead. Once a 
new value is available in the queue, it is removed and passed along to pyautogui via the ctrl key and a series of 
press commands.

The buttons are then constructed to populate our user interface. The list of characters are produced for the button 
layout and a callback handler is defined for responding to the click event. When a button is clicked on, the index of 
the character list is grabbed, incremented by one and added to the queue.

Finally, a thread is started to listen for items being added to the queue and a loop is started that serves as an 
entry point for all Tkinter events.

Usage:
To use this app, a user interacts directly with the Tkinter graphical user interface. When selecting a button, 
their desired hotkey is sent to pyautogui via the queue and thread structure. The shortcut is then activated, 
depending on your context."""


root = tk.Tk()
q = queue.Queue()


def spam_key(q):
    value = "1"
    while True:
        if not q.empty():
            value = q.get()
        else:
            if value == 10:
                continue
            pyautogui.keyDown('ctrl')
            pyautogui.press(str(value))
            pyautogui.keyUp('ctrl')
            time.sleep(0.05)


# create a list of characters for our buttons
chars = ['Title', 'Explanation', 'Answer', 'SuTitle', 'Header', 'Footer', "Heading", "Starting", "Ending", "Stop"]

# set the title of the window
root.title("Shortcuts App")


def handle_click(key):
    q.put(chars.index(key) + 1)


# create the buttons accordingly
for char in chars:
    btn = tk.Button(root, text=char, command=lambda c=char: handle_click(c))
    btn.pack()

# finally, start the main loop
t = threading.Thread(target=spam_key, args=(q,))
t.start()
root.mainloop()
