import tkinter as tk
from GUI import NanoQuantGUI  # Make sure GUI.py contains class NanoQuantGUI

def main():
    root = tk.Tk()
    app = NanoQuantGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
