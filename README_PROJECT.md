# Crop Disease Detection — Project scaffold

Overview
- Starter scaffold for training an image classifier to detect crop diseases using TensorFlow/Keras and transfer learning (MobileNetV2).

Structure
- `src/`: code (dataset, model, train, predict)
- `notebooks/`: starter notebook
- `models/`: saved models (gitignored)
- `data/`: place your `train/` and `val/` directories here

Quick start
1. Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare dataset directory with this layout:

```
data/train/<class_name>/*.jpg
data/val/<class_name>/*.jpg
```

3. Train:

```bash
python -m src.train --train-dir data/train --val-dir data/val --epochs 10 --save-path models/crop_model.h5
```

4. Predict:

```bash
python -m src.predict --model models/crop_model.h5 --image path/to/sample.jpg
```

Next steps I can help with
- Add dataset download script, augmentations, or conversion
- Add evaluation and visualization utilities
- Create a Colab notebook or Dockerfile
- Implement active learning or explainability

Dataset preparation helper
- Use the included script to download/extract and split datasets into `data/train` and `data/val`:

```bash
# From project root: use a local extracted folder
python scripts/prepare_dataset.py --source-dir path/to/extracted_dataset --out data --val-split 0.2

# Or download a zip and prepare (replace URL with a dataset zip)
python scripts/prepare_dataset.py --download-url https://example.com/dataset.zip --out data --val-split 0.2
```

The script will detect class subdirectories and create `data/train/<class>` and `data/val/<class>`.

Docker / containerized inference
- Build the Docker image:

```bash
docker build -t crop-disease-inference:latest .
```

- Run a container (it expects `models/crop_model.h5` and `models/class_indices.json` to exist in the repo):

```bash
docker run --rm -p 8080:8080 -v $(pwd)/models:/app/models crop-disease-inference:latest
```

- Send an image to the server:

```bash
curl -X POST "http://localhost:8080/predict" -F "file=@/path/to/image.jpg"
```

Evaluation and visualization
- Use the evaluation script to compute metrics and plots on your `data/val` set:

```bash
python -m src.evaluate --model models/crop_model.h5 --val-dir data/val --out-dir evaluation
```

This will create `evaluation/classification_report.json`, `evaluation/confusion_matrix.png`, and `evaluation/sample_predictions.png`.

Or open the demo notebook: [notebooks/1-evaluation.ipynb](notebooks/1-evaluation.ipynb)

