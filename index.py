
"""Train and use a character-level Shakespeare text generator."""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, scrolledtext, ttk
from typing import Any, Callable, Optional


BASE_DIR = Path(__file__).resolve().parent
TEXT_PATH = BASE_DIR / "text" / "shakespeare.txt"
MODEL_DIR = BASE_DIR / "models"
WEIGHTS_PATH = MODEL_DIR / "shakespeare.weights.h5"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LEGACY_WEIGHTS_PATH = BASE_DIR / "model.h5"
LEGACY_CHECKPOINT_DIR = BASE_DIR / "training_checkpoints_LSTM"

SEQUENCE_LENGTH = 150
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
RNN_UNITS = 1024


def import_tensorflow() -> Any:
  """Import TensorFlow only when the user starts a model operation."""
  try:
    import tensorflow as tf
  except ImportError as error:
    raise RuntimeError(
      "TensorFlow is not installed. Install it with: "
      "python -m pip install tensorflow"
    ) from error
  return tf


def load_training_text() -> str:
  """Read and validate the training corpus."""
  if not TEXT_PATH.exists():
    raise FileNotFoundError(f"Training text was not found: {TEXT_PATH}")
  text = TEXT_PATH.read_text(encoding="utf-8")
  if len(text) <= SEQUENCE_LENGTH:
    raise ValueError("The training text is too short for the configured sequence length.")
  return text


def build_model(tf: Any, vocab_size: int, batch_size: int) -> Any:
  """Create the stateful LSTM used for training and text generation."""
  return tf.keras.Sequential(
    [
      tf.keras.layers.Embedding(
        vocab_size,
        EMBEDDING_DIM,
        batch_input_shape=[batch_size, None],
      ),
      tf.keras.layers.LSTM(
        RNN_UNITS,
        return_sequences=True,
        stateful=True,
        recurrent_initializer="glorot_uniform",
      ),
      tf.keras.layers.Dense(vocab_size),
    ]
  )


def prepare_dataset(tf: Any, text: str) -> tuple[Any, list[str], dict[str, int]]:
  """Convert the corpus into shuffled input and target sequences."""
  vocabulary = sorted(set(text))
  char_to_index = {character: index for index, character in enumerate(vocabulary)}
  encoded_text = [char_to_index[character] for character in text]

  character_dataset = tf.data.Dataset.from_tensor_slices(encoded_text)
  sequences = character_dataset.batch(SEQUENCE_LENGTH + 1, drop_remainder=True)

  def split_sequence(chunk: Any) -> tuple[Any, Any]:
    return chunk[:-1], chunk[1:]

  dataset = sequences.map(split_sequence)
  dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
  return dataset, vocabulary, char_to_index


def compile_model(tf: Any, model: Any) -> None:
  """Compile the model with sparse categorical cross-entropy."""
  model.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  )


def find_existing_weights() -> Optional[str]:
  """Return the newest usable weights file or checkpoint, if available."""
  if WEIGHTS_PATH.exists():
    return str(WEIGHTS_PATH)
  if LEGACY_WEIGHTS_PATH.exists():
    return str(LEGACY_WEIGHTS_PATH)

  try:
    tf = import_tensorflow()
    checkpoint = tf.train.latest_checkpoint(str(CHECKPOINT_DIR))
    if checkpoint is None:
      checkpoint = tf.train.latest_checkpoint(str(LEGACY_CHECKPOINT_DIR))
  except RuntimeError:
    checkpoint = None
  return checkpoint


def generate_text(
  tf: Any,
  model: Any,
  prompt: str,
  vocabulary: list[str],
  char_to_index: dict[str, int],
  length: int,
  temperature: float,
) -> str:
  """Generate text from a prompt using the trained model."""
  if not prompt:
    raise ValueError("Enter a starting prompt first.")
  unsupported = sorted(set(character for character in prompt if character not in char_to_index))
  if unsupported:
    raise ValueError(f"The prompt contains unsupported characters: {unsupported!r}")

  input_eval = tf.expand_dims([char_to_index[character] for character in prompt], 0)
  generated_characters: list[str] = []
  model.reset_states()

  for _ in range(length):
    predictions = model(input_eval)
    predictions = tf.squeeze(predictions, 0) / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    generated_characters.append(vocabulary[predicted_id])

  return prompt + "".join(generated_characters)


class ShakespeareApp:
  """Small desktop interface for training and generating text."""

  def __init__(self, root: tk.Tk) -> None:
    self.root = root
    self.root.title("Shakespeare Text Generator")
    self.root.geometry("900x650")
    self.root.minsize(700, 500)
    self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
    self.model: Any = None
    self.tensorflow: Any = None
    self.vocabulary: list[str] = []
    self.char_to_index: dict[str, int] = {}
    self.is_busy = False

    self.prompt_var = tk.StringVar(value="ROMEO:")
    self.length_var = tk.IntVar(value=400)
    self.temperature_var = tk.DoubleVar(value=0.7)
    self.epochs_var = tk.IntVar(value=10)
    self.status_var = tk.StringVar(value="Ready. Load an existing model or train a new one.")

    self.create_widgets()
    self.root.after(100, self.process_events)

  def create_widgets(self) -> None:
    """Build the application controls."""
    self.root.columnconfigure(0, weight=1)
    self.root.rowconfigure(1, weight=1)

    controls = ttk.LabelFrame(self.root, text="Model controls", padding=12)
    controls.grid(row=0, column=0, sticky="ew", padx=12, pady=12)
    controls.columnconfigure(1, weight=1)

    ttk.Label(controls, text="Prompt").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=4)
    ttk.Entry(controls, textvariable=self.prompt_var).grid(row=0, column=1, sticky="ew", pady=4)
    ttk.Label(controls, text="Length").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
    ttk.Spinbox(controls, from_=1, to=5000, textvariable=self.length_var, width=10).grid(
      row=1, column=1, sticky="w", pady=4
    )
    ttk.Label(controls, text="Temperature").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
    ttk.Scale(controls, from_=0.1, to=1.5, variable=self.temperature_var, orient="horizontal").grid(
      row=2, column=1, sticky="ew", pady=4
    )
    ttk.Label(controls, text="Epochs for training").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=4)
    ttk.Spinbox(controls, from_=1, to=1000, textvariable=self.epochs_var, width=10).grid(
      row=3, column=1, sticky="w", pady=4
    )

    buttons = ttk.Frame(controls)
    buttons.grid(row=0, column=2, rowspan=4, padx=(20, 0))
    self.load_button = ttk.Button(buttons, text="Load model", command=self.load_model)
    self.load_button.grid(row=0, column=0, sticky="ew", pady=3)
    self.train_button = ttk.Button(buttons, text="Train model", command=self.train_model)
    self.train_button.grid(row=1, column=0, sticky="ew", pady=3)
    self.generate_button = ttk.Button(buttons, text="Generate text", command=self.generate)
    self.generate_button.grid(row=2, column=0, sticky="ew", pady=3)

    output_frame = ttk.LabelFrame(self.root, text="Generated text", padding=8)
    output_frame.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))
    output_frame.columnconfigure(0, weight=1)
    output_frame.rowconfigure(0, weight=1)
    self.output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, font=("Consolas", 11))
    self.output.grid(row=0, column=0, sticky="nsew")

    ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor="w").grid(
      row=2, column=0, sticky="ew", padx=12, pady=(0, 8)
    )

  def set_busy(self, busy: bool) -> None:
    """Enable or disable actions while a background operation runs."""
    self.is_busy = busy
    state = "disabled" if busy else "normal"
    for button in (self.load_button, self.train_button, self.generate_button):
      button.configure(state=state)

  def run_in_background(self, action: Callable[[], Any]) -> None:
    """Run TensorFlow work without blocking the Tk event loop."""
    if self.is_busy:
      return
    self.set_busy(True)
    threading.Thread(target=self.background_worker, args=(action,), daemon=True).start()

  def background_worker(self, action: Callable[[], Any]) -> None:
    try:
      result = action()
      self.events.put(("success", result))
    except Exception as error:
      self.events.put(("error", str(error)))

  def process_events(self) -> None:
    """Apply background results on the Tk main thread."""
    try:
      event, value = self.events.get_nowait()
    except queue.Empty:
      self.root.after(100, self.process_events)
      return

    self.set_busy(False)
    if event == "error":
      self.status_var.set("Operation failed")
      messagebox.showerror("Error", value)
    else:
      self.status_var.set(value)
    self.root.after(100, self.process_events)

  def prepare_runtime(self) -> tuple[Any, str]:
    """Load TensorFlow and prepare vocabulary data."""
    self.tensorflow = self.tensorflow or import_tensorflow()
    text = load_training_text()
    _, self.vocabulary, self.char_to_index = prepare_dataset(self.tensorflow, text)
    return self.tensorflow, text

  def load_model(self) -> None:
    """Load saved weights and skip training when a model already exists."""
    self.status_var.set("Loading model...")
    self.run_in_background(self._load_model)

  def _load_model(self) -> str:
    tf, _ = self.prepare_runtime()
    weights_path = find_existing_weights()
    if not weights_path:
      raise FileNotFoundError("No saved model was found. Use Train model first.")
    self.model = build_model(tf, len(self.vocabulary), batch_size=1)
    compile_model(tf, self.model)
    self.model.load_weights(weights_path)
    return f"Loaded model: {Path(weights_path).name}"

  def train_model(self) -> None:
    """Train a model and save its final weights."""
    self.status_var.set("Training started. This can take a while...")
    self.run_in_background(self._train_model)

  def _train_model(self) -> str:
    tf, text = self.prepare_runtime()
    dataset, _, _ = prepare_dataset(tf, text)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    model = build_model(tf, len(self.vocabulary), batch_size=BATCH_SIZE)
    compile_model(tf, model)
    checkpoint_path = str(CHECKPOINT_DIR / "epoch-{epoch:02d}")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_path,
      save_weights_only=True,
    )
    model.fit(dataset, epochs=self.epochs_var.get(), callbacks=[checkpoint_callback])
    model.save_weights(str(WEIGHTS_PATH))
    self.model = build_model(tf, len(self.vocabulary), batch_size=1)
    compile_model(tf, self.model)
    self.model.load_weights(str(WEIGHTS_PATH))
    return f"Training complete. Saved model to {WEIGHTS_PATH.name}."

  def generate(self) -> None:
    """Generate text with the loaded model."""
    if self.model is None:
      self.status_var.set("Load or train a model first.")
      messagebox.showinfo("Model required", "Load an existing model or train a new one first.")
      return
    try:
      result = generate_text(
        self.tensorflow,
        self.model,
        self.prompt_var.get(),
        self.vocabulary,
        self.char_to_index,
        self.length_var.get(),
        self.temperature_var.get(),
      )
      self.output.delete("1.0", tk.END)
      self.output.insert(tk.END, result)
      self.status_var.set("Text generated.")
    except Exception as error:
      messagebox.showerror("Generation error", str(error))


def main() -> None:
  root = tk.Tk()
  app = ShakespeareApp(root)
  if WEIGHTS_PATH.exists() or LEGACY_WEIGHTS_PATH.exists():
    root.after(200, app.load_model)
  root.mainloop()


if __name__ == "__main__":
  main()
