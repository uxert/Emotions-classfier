import PySimpleGUI as sg
from EmotionRecognizer import EmotionRecognizer
from RAVDESSDataset import RAVDESSDataset
from playsound import playsound

class MyGUI:

    def __init__(self):
        self.correct = 0
        self.all_preds = 0
        self.current_audio_path = None
        self.DEFAULT_FONT = ("Helvetica", 16)

    def next_sample(self, window, model: EmotionRecognizer, dataset: RAVDESSDataset, class_names: list[str]) -> None:
        true_emotion, predicted_emotion, audio_path = model.make_one_prediction(dataset, class_names)
        window["-PRED-"].update(f"Prediction: {predicted_emotion}")
        window["-TRUE-"].update(f"True emotion: {true_emotion}")
        self.all_preds += 1
        if true_emotion == predicted_emotion:
            self.correct += 1
        self.current_audio_path = audio_path
        window["-FILEPATH-"].update(f"Audio path: {self.current_audio_path}")
        window['-RATIO-'].update(f"Correct predictions: {self.correct}\t| All predictions: {self.all_preds}")

    def display_current_audio(self):
        playsound(self.current_audio_path)

    def show_gui(self, model, dataset, class_names):

        # Layout of the GUI
        layout = [
            [sg.Text("Prediction: None (yet...)", key="-PRED-", size=(30, 1))],
            [sg.Text("True emotion: None (yet...) ", key="-TRUE-", size=(30, 1))],
            [sg.Text("Audio path: None (yet...) ", key="-FILEPATH-", size=(100, 1))],
            [sg.Text("Correct predictions: 0\t| All predictions: 0", key="-RATIO-", size=(45, 1))],
            [sg.Button("Next sample", key="-BTN1-"), sg.Button("Play audio", key="-BTN2-"),
             sg.Button("Exit")]
        ]


        # Create the Window
        window = sg.Window("Emotion classifier", layout, size=(1200, 250), font=self.DEFAULT_FONT)

        # Event Loop
        while True:
            event, values = window.read()

            # If user closes window or clicks Exit
            if event == sg.WINDOW_CLOSED or event == "Exit":
                break

            # Button Actions
            if event == "-BTN1-":
                self.next_sample(window, model, dataset, class_names)
            elif event == "-BTN2-":
                self.display_current_audio()

        # Close the Window
        window.close()