import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class Scroll(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.canvas = tk.Canvas(self, bg="#c3c1c1", width=180, height=500)
        self.scrollable_frame = ttk.Frame(self.canvas)
        
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        
        self.scrollable_frame.bind("<Configure>",lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        self.items = []

    def add_item(self, crop, plate):
        "Add a item to list"
        
        img = Image.fromarray(crop)
        img.thumbnail((150, 50))
        img_tk = ImageTk.PhotoImage(image=img)
        
        item_canvas = tk.Canvas(self.scrollable_frame, width=180, height=50)
        item_canvas.create_image(10, 10, anchor="nw", image=img_tk)        
        item_canvas.image = img_tk
        item_canvas.pack(fill="x", pady=5)
                
        item_canvas.create_text(110, 25, text=plate, fill="red", font=('Arial', 12, 'bold'))
        
        self.items.append(item_canvas)

        if len(self.items) > 8:
            self.canvas.configure(scrollregion=self.canvas.bbox("all"))
            self.scrollbar.pack(side="right", fill="y")
        else:
            self.scrollbar.pack_forget()
            
    def clear(self):
        self.items.clear()        
        self.scrollbar.pack_forget()
                