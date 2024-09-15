import tkinter as tk
from PIL import Image, ImageTk

class Video:
    def __init__(self, parent):
        self.video_streaming_box = tk.Canvas(parent, width=1200, height=650, bg="#c3c1c1")
        self.video_streaming_box.grid(row=0, column=0, columnspan=2, sticky="n")
        self.set_status_video_box("Initial")
        
    def update_frame(self, frame):
        "Update the canvas with new frame"
            
        img = Image.fromarray(frame)
        img.thumbnail((1200, 800))
        img_tk = ImageTk.PhotoImage(image=img)

        if self.video_streaming_box.getvar("ClearVideoBox") != "Clean":
            self.video_streaming_box.create_image(0, 0, anchor="nw", image=img_tk)
            self.video_streaming_box.image = img_tk
            
    def clear(self):        
        self.video_streaming_box.delete("all")
        self.video_streaming_box.image = None
        self.set_status_video_box("Clean")
        
    def set_status_video_box(self, status):        
        self.video_streaming_box.setvar("ClearVideoBox", status)
        
    def get_status_video_box(self):        
        self.video_streaming_box.getvar("ClearVideoBox")
        
    def schedule_playback(self, interval, callback):
        "Execution of the callback function every time interval"
        self.video_streaming_box.after(interval, callback)
