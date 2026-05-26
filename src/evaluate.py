import argparse
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report
from src.visualize import plot_confusion_matrix, plot_sample_predictions


def create_val_generator(val_dir, img_size=(224, 224), batch_size=32):
    datagen = ImageDataGenerator(rescale=1.0/255.0)
    gen = datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
    )
    return gen


def evaluate(model_path, val_dir, img_size=224, batch_size=32, out_dir='evaluation'):
    os.makedirs(out_dir, exist_ok=True)
    model = load_model(model_path)

    # try load class indices from same dir
    class_indices_path = os.path.join(os.path.dirname(model_path), 'class_indices.json')
    class_map = None
    if os.path.exists(class_indices_path):
        with open(class_indices_path, 'r') as fh:
            class_map = json.load(fh)

    gen = create_val_generator(val_dir, img_size=(img_size, img_size), batch_size=batch_size)

    steps = int(np.ceil(gen.samples / batch_size))
    preds = model.predict(gen, steps=steps)
    y_pred = np.argmax(preds, axis=1)
    y_true = gen.classes

    labels = [None] * len(gen.class_indices)
    for k, v in gen.class_indices.items():
        labels[v] = k

    report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
    with open(os.path.join(out_dir, 'classification_report.json'), 'w') as fh:
        json.dump(report, fh, indent=2)

    # confusion matrix plot
    plt = plot_confusion_matrix(y_true, y_pred, labels)
    plt.savefig(os.path.join(out_dir, 'confusion_matrix.png'))
    plt.close()

    # sample predictions
    try:
        plt2 = plot_sample_predictions(model, gen, class_map or gen.class_indices, num=6, img_size=img_size)
        plt2.savefig(os.path.join(out_dir, 'sample_predictions.png'))
        plt2.close()
    except Exception as e:
        print('Could not create sample predictions plot:', e)

    print('Saved evaluation artifacts to', out_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--out-dir', default='evaluation')
    args = parser.parse_args()
    evaluate(args.model, args.val_dir, img_size=args.img_size, batch_size=args.batch_size, out_dir=args.out_dir)
