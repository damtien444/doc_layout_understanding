import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk

# create a list of texts to visualize
texts = ["Text Sample 1", "Text Sample 2", "Text Sample 3"]

# create a list of labels for each text sample
labels = [0, 1, 0]

# create a dictionary to map label classes to colors for visualization
label2color = {0: 'blue', 1: 'red'}

# create a scatter plot with texts as points on x-axis,
# assigned color according to their label on y-axis
plt.scatter(x=range(len(texts)), y=[0] * len(texts), c=[label2color[label] for label in labels])

# customize the plot labels and title
plt.title("Text Visualization")
plt.xlabel("Text Samples")
plt.xticks(range(len(texts)), texts)
plt.yticks([])

# display the plot
plt.show()

# create a GUI window using tkinter
root = tk.Tk()

# set the window title
root.title("Label App")

# add a label to the window with the first text sample
text_label = tk.Label(root, text=texts[0])
text_label.pack()

# add radio buttons to the window for each label class
var = tk.IntVar()
for label_class in sorted(set(labels)):
    rb = tk.Radiobutton(root, text=f"Label {label_class}", variable=var, value=label_class)
    rb.pack()


# create a submit button that updates the label list with the selected label class
def submit():
    label = var.get()
    labels[0] = label

    # update the text label and clear the radio buttons
    text_label.config(text=texts[0] + f" (Label: {label})")
    var.set(0)


submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.pack()

# start the tkinter event loop
root.mainloop()
