import tkinter as tk
from GUI import NanoCount  # Make sure GUI.py contains class NanoCount

def main():
    root = tk.Tk()
    app = NanoCount(root)
    root.mainloop()

if __name__ == "__main__":
    main()
