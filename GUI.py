import tkinter as tk
from tkinter import ttk
from emotions_classifier.EmotionRecognizer import EmotionRecognizer
from emotions_classifier.RAVDESSDataset import RAVDESSDataset
from playsound import playsound
import threading  # to make the audio playback ascync

# Color constants
BG_COLOR = "#f0f0f0"        # Light gray background
ACCENT_BLUE = "#2980b9"     # Royal blue for important text
TEXT_GRAY = "#7f8c8d"       # Gray for secondary text
BUTTON_NEXT = "#3498db"
BUTTON_NEXT_HOVER = "#2980b9"
BUTTON_PLAY = "#2ecc71"
BUTTON_PLAY_HOVER = "#27ae60"
BUTTON_EXIT = "#e74c3c"
BUTTON_EXIT_HOVER = "#c0392b"

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
        self.stats_label = None

    def next_sample(self, model, dataset, class_names):
        true_emotion, predicted_emotion, audio_path = model.make_one_prediction(dataset, class_names)
        self.pred_label.config(text=f" {predicted_emotion}")
        self.true_label.config(text=f" {true_emotion}")
        self.all_preds += 1
        if true_emotion == predicted_emotion:
            self.correct += 1
        self.current_audio_path = audio_path
        self.filepath_label.config(text=f"Audio path: {self.current_audio_path}")
        self.stats_label.config(text=f"Correct: {self.correct} | Total: {self.all_preds} | "
                                     f"Accuracy: {self.correct / self.all_preds * 100:.2f}%")

    def async_display_current_audio(self):
        if self.current_audio_path:
            audio_thread = threading.Thread(target=playsound, args=[self.current_audio_path], daemon=True)
            audio_thread.start()

    def _create_main_frame(self) -> tk.Frame:
        main_frame = ttk.Frame(self.root, style="TFrame", padding=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        header = ttk.Label(main_frame, text="Emotion Recognition System", style="Header.TLabel")
        header.pack(anchor="w", pady=(0, 20))
        return main_frame

    @staticmethod
    def _setup_style():
        """Configures ttk styles"""
        style = ttk.Style()
        style.configure("TFrame", background=BG_COLOR)
        style.configure("TButton", font=("Helvetica", 12), padding=6)
        style.configure("TLabel", font=("Helvetica", 14), background=BG_COLOR)

        style.configure("Header.TLabel", font=("Helvetica", 20, "bold"), background=BG_COLOR)
        style.configure("Value.TLabel", font=("Helvetica", 18, "bold"), foreground=ACCENT_BLUE, background=BG_COLOR)
        style.configure("Path.TLabel", font=("Helvetica", 16), foreground=TEXT_GRAY, background=BG_COLOR)
        style.configure("Stats.TLabel", font=("Helvetica", 18))

    def _create_prediction_area(self, main_frame):
        """Creates the area where prediction and true value are displayed"""
        pred_frame = tk.Frame(main_frame, bg="white", bd=1, relief=tk.GROOVE, padx=15, pady=15)
        pred_frame.pack(fill=tk.X, pady=10)
        # Prediction row
        pred_row = tk.Frame(pred_frame, bg="white")
        pred_row.pack(fill=tk.X, pady=5)
        ttk.Label(pred_row, text="Prediction:", background="white", font=("Helvetica", 16)).pack(side=tk.LEFT, padx=(0, 10))
        self.pred_label = ttk.Label(pred_row, text="None (yet...)", style="Value.TLabel", background="white")
        self.pred_label.pack(side=tk.LEFT)

        # True emotion row
        true_row = tk.Frame(pred_frame, bg="white")
        true_row.pack(fill=tk.X, pady=5)
        ttk.Label(true_row, text="True emotion:", background="white", font=("Helvetica", 16)).pack(side=tk.LEFT, padx=(0, 10))
        self.true_label = ttk.Label(true_row, text="None (yet...)", style="Value.TLabel", background="white")
        self.true_label.pack(side=tk.LEFT)

    def _create_stats_frame(self, main_frame):
        stats_frame = tk.Frame(main_frame, bg="white", bd=1, relief=tk.GROOVE, padx=15, pady=10)
        stats_frame.pack(fill=tk.X, pady=10)
        self.stats_label = ttk.Label(stats_frame,
                                     text="Correct: 0 | Total: 0 | Accuracy: 00.00%",
                                     background="white", style="Stats.TLabel")
        self.stats_label.pack()

    def _create_filepath_label(self, main_frame):
        self.filepath_label = ttk.Label(main_frame, text="Audio path: None (yet...)", style="Path.TLabel")
        self.filepath_label.pack(fill=tk.X, pady=10)

    @staticmethod
    def _on_enter(button, hover_color):
        button['background'] = hover_color  # Darker color on hover

    @staticmethod
    def _on_leave(button, original_color):
        button['background'] = original_color  # Return to original color

    def _setup_buttons(self, main_frame, model: EmotionRecognizer, dataset: RAVDESSDataset, class_names):
        """
        Requires self._on_enter() and self._on_leave() methods to exist.
        Setups buttons, who would have thought...
        """
        button_frame = ttk.Frame(main_frame, style="TFrame")
        button_frame.pack(pady=10)
        buttons_font = ("Helvetica", 20)
        next_btn = tk.Button(button_frame, text="Next Sample", font=buttons_font,
                             bg=BUTTON_NEXT, fg="white", padx=10, pady=5,
                             command=lambda: self.next_sample(model, dataset, class_names))
        next_btn.pack(side=tk.LEFT, padx=10)
        play_btn = tk.Button(button_frame, text="Play Audio", font=buttons_font,
                             bg=BUTTON_PLAY, fg="white", padx=10, pady=5,
                             command=self.async_display_current_audio)
        play_btn.pack(side=tk.LEFT, padx=10)
        exit_btn = tk.Button(button_frame, text="Exit", font=buttons_font,
                             bg=BUTTON_EXIT, fg="white", padx=10, pady=5,
                             command=self.root.destroy)
        exit_btn.pack(side=tk.LEFT, padx=10)
        next_btn.bind("<Enter>", lambda e: self._on_enter(next_btn, BUTTON_NEXT))
        next_btn.bind("<Leave>", lambda e: self._on_leave(next_btn, BUTTON_NEXT_HOVER))
        play_btn.bind("<Enter>", lambda e: self._on_enter(play_btn, BUTTON_PLAY))
        play_btn.bind("<Leave>", lambda e: self._on_leave(play_btn, BUTTON_PLAY_HOVER))
        exit_btn.bind("<Enter>", lambda e: self._on_enter(exit_btn, BUTTON_EXIT))
        exit_btn.bind("<Leave>", lambda e: self._on_leave(exit_btn, BUTTON_EXIT_HOVER))

    def show_gui(self, model: EmotionRecognizer, dataset: RAVDESSDataset, class_names):
        self.root = tk.Tk()
        self.root.title("Emotion Classifier")
        self.root.geometry("1200x550")
        self.root.configure(bg=BG_COLOR)
        self._setup_style()

        main_frame = self._create_main_frame()

        self._create_prediction_area(main_frame)
        self._create_stats_frame(main_frame)
        self._create_filepath_label(main_frame)
        self._setup_buttons(main_frame, model, dataset, class_names)

        # Start the main event loop
        self.root.mainloop()
