import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from PIL import Image, ImageTk
from tkinter import ttk

def get_image_files(folder):
    images = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith((".jpg", ".png")):
                images.append(os.path.join(root, file))

    images.sort()
    return images

target_folder = "/home/tiendq/Desktop/DocRec/2_data_preparation/2_sample_selected"

class ImageViewer:
    def __init__(self, root, images):
        self.root = root
        self.images = images
        self.index = 0
        self.history = []

        self.label = tk.Label(self.root, width=800, height=800)
        self.label.pack()
        self.previous_button = tk.Button(self.root, text="Previous", command=self.previous)
        self.previous_button.pack()

        self.next_button = tk.Button(self.root, text="Next", command=self.next)
        self.next_button.pack()

        self.select_button = tk.Button(self.root, text="Select", command=self.select)
        self.select_button.pack()

        self.file_name = tk.StringVar()
        self.file_entry = tk.Entry(self.root, textvariable=self.file_name, width=150)
        self.file_entry.pack()

        self.text = tk.Label(self.root, text="Just do it")
        self.text.pack()

        self.progress = ttk.Progressbar(self.root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack()

        self.last_choices = tk.Listbox(self.root, height=50, width=150)
        self.last_choices.pack()


        self.root.bind("<Left>", self.previous)
        self.root.bind("<Right>", self.next)
        self.root.bind("<Return>", self.select)
        # self.update_image()

    def previous(self, event=None):
        self.index = (self.index - 1) % len(self.images)

        self.update_image()

    def next(self, event=None):
        self.index = (self.index + 1) % len(self.images)
        self.update_image()

    def select(self, event=None):
        image = self.images[self.index]
        os.system(f'cp "{image}" {target_folder}')
        # messagebox.showinfo("Selected Image", f"You selected {image}")
        print(f"You selected {image}")
        self.history.append(image)
        self.last_choices.delete(0, tk.END)
        for last_choice in self.history[-10:]:
            self.last_choices.insert(0, last_choice.split("/")[-1] + "\n")


    def update_image(self):
        self.file_name.set(self.images[self.index].split("/")[-1])
        image = Image.open(self.images[self.index])

        selected_list = os.listdir(target_folder)
        self.text.configure(text=str(len(selected_list))+"------"+str(self.index + 1)+"/"+str(len(self.images)))
        self.progress['value'] = (self.index + 1) / len(self.images) * 100
        ratio = min(self.label.winfo_width() / image.width, self.label.winfo_height() / image.height)
        new_width = int(image.width * ratio)
        new_height = int(image.height * ratio)
        image = image.resize((new_width, new_height), Image.ANTIALIAS)
        photo = ImageTk.PhotoImage(image)
        self.label.config(image=photo)
        self.label.image = photo


if __name__ == "__main__":
    root = tk.Tk()
    folder = filedialog.askdirectory(title="Select Folder")
    images = get_image_files(folder)
    viewer = ImageViewer(root, images)
    root.mainloop()
