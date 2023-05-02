import tkinter as tk

class WhiteboardApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Whiteboard App")
        
        # Create canvas for drawing
        self.canvas = tk.Canvas(self.root, width=600, height=400, bg="white")
        self.canvas.pack(expand=True, fill=tk.BOTH)
        self.canvas.bind("<B1-Motion>", self.draw)
        
        # Create shape buttons
        self.circle_btn = tk.Button(self.root, text="Circle", command=self.draw_circle)
        self.circle_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.triangle_btn = tk.Button(self.root, text="Triangle", command=self.draw_triangle)
        self.triangle_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.polygon_btn = tk.Button(self.root, text="Polygon", command=self.draw_polygon)
        self.polygon_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Create menu
        self.menu = tk.Menu(self.root)
        self.root.config(menu=self.menu)
        file_menu = tk.Menu(self.menu)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save", command=self.save)
        file_menu.add_command(label="Open", command=self.open)
        
        self.root.mainloop()
    
    def draw(self, event):
        # Draw a line on the canvas
        x, y = event.x, event.y
        self.canvas.create_line(x, y, event.x+1, event.y+1, fill="black", width=2)
    
    def draw_circle(self):
        # Show dialog box for circle parameters
        # and draw circle on canvas
        pass
    
    def draw_triangle(self):
        # Show dialog box for triangle parameters
        # and draw triangle on canvas
        pass
    
    def draw_polygon(self):
        # Show dialog box for polygon parameters
        # and draw polygon on canvas
        pass
    
    def save(self):
        # Save drawing to file
        pass
    
    def open(self):
        # Load drawing from file
        pass

if __name__ == "__main__":
    app = WhiteboardApp()
