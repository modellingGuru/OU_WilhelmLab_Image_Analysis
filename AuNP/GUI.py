import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
from PIL import Image, ImageTk
import numpy as np
import cv2
import os
import webbrowser

from metrics import calculate_density, compute_nnd
from visualization import plot_nnd_histogram, plot_3d_scatter
from segmentation import segment_nanoparticles
from thresholding import apply_threshold
from preprocessing import preprocess_image

class NanoCount:
    def __init__(self, root):
        self.root = root
        self.root.title("🌟 NanoCount - Nanoparticle Quantification Tool 🌟")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f4ff")

        self.image = None
        self.create_widgets()

    def create_widgets(self):
        title_label = tk.Label(self.root, text="NanoCount - Nanoparticle Quantification Tool",
                               font=("Helvetica", 18, "bold"), bg="#4a90e2", fg="white", pady=10)
        title_label.pack(fill="x")

        # Button Panel
        button_frame = tk.Frame(self.root, bg="#f0f4ff")
        button_frame.pack(pady=10)

        tk.Button(button_frame, text="📷 Upload Image", command=self.upload_image,
                  bg="#66bb6a", fg="white", font=("Helvetica", 10, "bold")).grid(row=0, column=0, padx=10)

        tk.Button(button_frame, text="📊 Upload CSV", command=self.upload_csv,
                  bg="#ffa726", fg="white", font=("Helvetica", 10, "bold")).grid(row=0, column=1, padx=10)
        
        tk.Button(button_frame, text="📁 Batch Process", command=self.batch_process,
          bg="#26c6da", fg="white", font=("Helvetica", 10, "bold")).grid(row=0, column=4, padx=10)

        tk.Button(button_frame, text="🔍 Quantify Particles", command=self.quantify_particles,
                  bg="#42a5f5", fg="white", font=("Helvetica", 10, "bold")).grid(row=0, column=2, padx=10)

        # Threshold/Parameter Entry
        param_frame = tk.Frame(self.root, bg="#f0f4ff")
        param_frame.pack(pady=5)

        tk.Label(param_frame, text="Threshold (0-255):", bg="#f0f4ff").grid(row=0, column=0)
        self.threshold_entry = tk.Entry(param_frame, width=10)
        self.threshold_entry.insert(0, "100")
        self.threshold_entry.grid(row=0, column=1, padx=5)

        tk.Label(param_frame, text="Min Particle Area (px):", bg="#f0f4ff").grid(row=0, column=2)
        self.min_area_entry = tk.Entry(param_frame, width=10)
        self.min_area_entry.insert(0, "50")
        self.min_area_entry.grid(row=0, column=3, padx=5)

        # Image Display
        self.image_label = tk.Label(self.root, bg="#e3f2fd")
        self.image_label.pack(pady=10)

        # Text Results
        result_frame = tk.LabelFrame(self.root, text="📈 Quantification Results",
                                     bg="#f0f4ff", fg="#333", padx=10, pady=10)
        result_frame.pack(padx=10, pady=10, fill="both", expand=False)

        self.result_text = tk.Text(result_frame, height=8, width=110, font=("Courier", 10))
        self.result_text.pack()

        # Plot Frame
        self.plot_frame = tk.LabelFrame(self.root, text="📊 Visualizations",
                                        bg="#f0f4ff", fg="#333", padx=10, pady=10)
        self.plot_frame.pack(padx=10, pady=10, fill="both", expand=True)
        
    def batch_process(self):
    folder = filedialog.askdirectory(title="Select Folder of Images")
    if not folder:
        return

    out_folder = filedialog.askdirectory(title="Select Output Folder")
    if not out_folder:
        return

    log_path = os.path.join(out_folder, "batch_summary.txt")
    with open(log_path, "w") as log:
        log.write("File,Particles Detected\n")

        for file in os.listdir(folder):
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                try:
                    path = os.path.join(folder, file)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    particles = segment_nanoparticles(img)
                    coords = particles
                    if coords.shape[1] == 2:
                        coords = np.column_stack([coords, np.zeros(len(coords))])
                    out_csv = os.path.join(out_folder, file.replace(".", "_") + "_particles.csv")
                    np.savetxt(out_csv, coords, delimiter=",", fmt="%.2f", header="x,y,z", comments='')
                    log.write(f"{file},{len(particles)}\n")
                except Exception as e:
                    log.write(f"{file},ERROR: {e}\n")

    messagebox.showinfo("Batch Complete", f"Batch processing complete.\nSummary saved to:\n{log_path}")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp")])
        if file_path:
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image(self.image)

    def upload_csv(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        try:
            coords = np.loadtxt(file_path, delimiter=",")
            if coords.ndim == 1:
                coords = np.expand_dims(coords, axis=0)

            # Ask for voxel size and shape
            shape_input = simpledialog.askstring("Image Shape", "Enter image shape (e.g. 512,512,100):")
            voxel_input = simpledialog.askstring("Voxel Size", "Enter voxel size in µm (e.g. 0.2,0.2,0.2):")

            image_shape = tuple(map(int, shape_input.split(',')))
            voxel_size = tuple(map(float, voxel_input.split(',')))

            count, volume, density = calculate_density(coords, image_shape, voxel_size)
            nnd = compute_nnd(coords)

            mean_nnd = np.mean(nnd)
            std_nnd = np.std(nnd)

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "📊 Nanoparticle CSV Metrics\n")
            self.result_text.insert(tk.END, f"Total Particles: {count}\n")
            self.result_text.insert(tk.END, f"Volume: {volume:.2f} µm³\n")
            self.result_text.insert(tk.END, f"Density: {density:.4f} particles/µm³\n")
            self.result_text.insert(tk.END, f"Mean NND: {mean_nnd:.2f} µm\n")
            self.result_text.insert(tk.END, f"Std NND: {std_nnd:.2f} µm\n")

            # Clear old plots
            for widget in self.plot_frame.winfo_children():
                widget.destroy()

            # Add plots
            plot_nnd_histogram(nnd, self.plot_frame)
            html_path = plot_3d_scatter(coords)
            webbrowser.open(os.path.abspath(html_path))

        except Exception as e:
            messagebox.showerror("CSV Error", f"Failed to process CSV: {e}")

    def display_image(self, img):
        resized = cv2.resize(img, (400, 400))
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        self.image_label.configure(image=img_tk)
        self.image_label.image = img_tk

    def quantify_particles(self):
        if self.image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        try:
            threshold_val = int(self.threshold_entry.get())
            min_area = int(self.min_area_entry.get())
        except ValueError:
            messagebox.showerror("Input Error", "Please enter valid numeric values.")
            return

        particles = find_particles(self.image, threshold_val, min_area)
        overlay = self.image.copy()
        cv2.drawContours(overlay, particles, -1, (255, 255, 255), 1)
        self.display_image(cv2.cvtColor(overlay, cv2.COLOR_GRAY2RGB))

        areas = [cv2.contourArea(c) for c in particles]
        avg_area = np.mean(areas) if areas else 0

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"Particles Detected: {len(particles)}\n")
        self.result_text.insert(tk.END, f"Average Area: {avg_area:.2f} px²\n")
