import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class Scroll(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title_label = tk.Label(self, text="Detections", font=("Arial", 12, "bold"))
        self.title_label.pack(side="top", pady=10)
        
        self.canvas = tk.Canvas(self, bg="#c3c1c1", width=180, height=500, bd=1, relief="solid")        
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
                
        self.createFrameIfDoesntExists()
        
        img = Image.fromarray(crop)
        img.thumbnail((150, 50))
        img_tk = ImageTk.PhotoImage(image=img)
        
        item_canvas = tk.Canvas(self.scrollable_frame, width=175, height=50)
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
        "Delete all items of list"
                
        for item_canvas in self.items:
            item_canvas.image = None
            item_canvas.delete("all")
            item_canvas.pack_forget()
        
        self.items.clear()

        # Restart the scrollBar
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        self.scrollbar.pack_forget()  # Hide the scrollbar if it doenst have items
        
        if not self.scrollable_frame is None:
            self.scrollable_frame.destroy()
            self.scrollable_frame = None
        
    def createFrameIfDoesntExists(self):        
        if self.scrollable_frame is None:
            self.scrollable_frame = ttk.Frame(self.canvas)
            self.scrollable_frame.bind("<Configure>",lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))        
            self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")