# Shakespeare Text Generator

A character-level Shakespeare text generator built with TensorFlow and a stateful LSTM network. The project includes a desktop GUI for training, loading saved weights, and generating new text from a custom prompt.

## Features

- Character-level text generation from Shakespeare's works
- Stateful LSTM architecture with configurable training epochs
- Automatic loading of an existing model when saved weights are available
- Support for legacy `model.h5` and `training_checkpoints_LSTM` files
- Background training so the desktop interface remains responsive
- Adjustable generation length and temperature
- Simple Tkinter desktop interface
- English-only source comments and interface text

## How It Works

1. The corpus in `text/shakespeare.txt` is converted into character indices.
2. The text is split into sequences of 150 characters.
3. An LSTM learns to predict the next character in each sequence.
4. The trained weights are saved in the `models` directory.
5. The model generates new text one character at a time from the supplied prompt.

## Requirements

- Python 3.8
- TensorFlow 2.10.1
- Tkinter, usually included with the standard Python installation

## Installation

Clone the repository and open its directory:

```bash
git clone <your-repository-url>
cd shakespeareGan
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Application

```bash
python index.py
```

### Using the GUI

- **Load model** loads the saved model without retraining.
- **Train model** trains a new model and saves its weights.
- **Generate text** creates text using the current prompt and settings.
- **Prompt** defines the starting text, for example `ROMEO:`.
- **Length** controls the number of generated characters.
- **Temperature** controls randomness. Lower values produce more predictable text.
- **Epochs for training** controls the training duration.

If a saved model exists, the application attempts to load it automatically at startup. If no model is available, train one from the GUI.

## Model Files

The current model is saved as:

```text
models/shakespeare.weights.h5
```

Training checkpoints are saved under:

```text
models/checkpoints/
```

The application also recognizes these legacy paths when they exist:

```text
model.h5
training_checkpoints_LSTM/
```

## Project Structure

```text
shakespeareGan/
├── index.py                 # GUI, training, loading, and generation logic
├── requirements.txt         # Python dependencies
├── text/
│   └── shakespeare.txt      # Training corpus
└── models/                  # Saved weights and checkpoints
```

## Notes

Training time depends on the available CPU/GPU and the selected number of epochs. The included corpus and model configuration are intended as an educational and experimental text-generation project rather than a production language model.

## License

No license has been selected for this repository yet. Add a license before distributing the project publicly.
