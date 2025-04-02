import tkinter as tk
from tkinter import font as tkfont  # Add this import
from EmotionRecognizer import EmotionRecognizer
from RAVDESSDataset import RAVDESSDataset
from playsound import playsound

class MyGUI:
    def __init__(self):
        self.correct = 0
        self.all_preds = 0
        self.current_audio_path = None
        self.DEFAULT_FONT = ("Helvetica", 16)
        self.root = None

        # References to UI elements we'll need to update
        self.pred_label = None
        self.true_label = None
        self.filepath_label = None
        self.ratio_label = None

    def next_sample(self, model, dataset, class_names):
        true_emotion, predicted_emotion, audio_path = model.make_one_prediction(dataset, class_names)
        self.pred_label.config(text=f"Prediction: {predicted_emotion}")
        self.true_label.config(text=f"True emotion: {true_emotion}")
        self.all_preds += 1
        if true_emotion == predicted_emotion:
            self.correct += 1
        self.current_audio_path = audio_path
        self.filepath_label.config(text=f"Audio path: {self.current_audio_path}")
        self.ratio_label.config(text=f"Correct predictions: {self.correct}\t| All predictions: {self.all_preds}")

    def display_current_audio(self):
        if self.current_audio_path:
            playsound(self.current_audio_path)

    def show_gui(self, model: EmotionRecognizer, dataset: RAVDESSDataset, class_names):
        self.root = tk.Tk()
        self.root.title("Emotion classifier")
        self.root.geometry("1200x250")

        # Set default font
        default_font = tkfont.nametofont("TkDefaultFont")
        default_font.configure(family="Helvetica", size=16)
        self.root.option_add("*Font", default_font)

        # Create UI elements
        self.pred_label = tk.Label(self.root, text="Prediction: None (yet...)", anchor="w", width=30)
        self.pred_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

        self.true_label = tk.Label(self.root, text="True emotion: None (yet...)", anchor="w", width=30)
        self.true_label.grid(row=1, column=0, sticky="w", padx=10, pady=5)

        self.filepath_label = tk.Label(self.root, text="Audio path: None (yet...)", anchor="w", width=100)
        self.filepath_label.grid(row=2, column=0, columnspan=3, sticky="w", padx=10, pady=5)

        self.ratio_label = tk.Label(self.root, text="Correct predictions: 0\t| All predictions: 0", anchor="w", width=45)
        self.ratio_label.grid(row=3, column=0, sticky="w", padx=10, pady=5)

        # Create button frame
        button_frame = tk.Frame(self.root)
        button_frame.grid(row=4, column=0, sticky="w", padx=10, pady=10)

        # Create buttons
        next_btn = tk.Button(button_frame, text="Next sample",
                            command=lambda: self.next_sample(model, dataset, class_names))
        next_btn.pack(side=tk.LEFT, padx=5)

        play_btn = tk.Button(button_frame, text="Play audio", command=self.display_current_audio)
        play_btn.pack(side=tk.LEFT, padx=5)

        exit_btn = tk.Button(button_frame, text="Exit", command=self.root.destroy)
        exit_btn.pack(side=tk.LEFT, padx=5)

        # Start the main event loop
        self.root.mainloop()