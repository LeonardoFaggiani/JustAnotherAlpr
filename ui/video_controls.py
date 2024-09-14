import tkinter as tk

class Button:
    def __init__(self, parent, buttonText, row, col, callback_function, width=15, padx=10):        
        self.button_frame = parent        
        
        self.button = tk.Button(self.button_frame, text=buttonText, width=width, command=callback_function)
        self.button.grid(row=row, column=col, padx=padx)

    def enable(self):
        "Enable button."
        self.button.config(state="normal")

    def disable(self):
        "Disable button."
        self.button.config(state="disabled")