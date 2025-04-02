# Emotion recognition from audio recordings

## What is it?
A little project, that creates, trains, and allows for inference of a PyTorch model recognizing emotion from audio.

This would not be possible without the [RAVDESS dataset](https://zenodo.org/record/1188976). 
There are 8 possible emotions in the dataset, and every sample is a recording of one from only two sentences. That means,
the model cannot determine the emotion based on words - it has to learn to analyze things like pitch and loudness

You can either use this setup to try to train your own model or use mine - it's posted on google drive and will download automatically
if you try to call inference and no model is detected in your directory.

Currently it has around 50% to 60% accuracy on test data. Given there are 8 different labels it's much better than random guessing.
And in some cases even I cannot classify these recordings correctly...


## How to use it?
### Setup
1. Clone this repository
2. After setting up venv, or any other environment you want to use, you need to `pip install -r requirements.txt`
3. Install pytorch and torchaudio. PyTorch can be setup in many ways, depending if you have CUDA, ROCm, etc... So it's not in requirements.txt.
I just recommend [PyTorch Start Locally](https://pytorch.org/get-started/locally/) instead.
The process is easy, just one command - but command specifically tailored to your system. You do not need torchvision.


### Usage
After setup, the usage is pretty straightforward:
- **run main.py to run inference**. It will download the model from my google drive, if it's not found in the same directory as main. Afterwards it'll launch a simple GUI that will
 allow you to see the model's predictions, true answers, and play the audio files.
- **run train.py to train your own model from scratch**. Afterwards you can run main.py to launch inference and experiment with the model. **This project is MIT Licensed, so feel free to experiment with it all you want!**


The data is automatically split to training and inference, so you can be certain, that during inference model is given data it has never seen before.
