import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def predict_image(model_path, img_path, img_size=224, class_indices=None):
    model = load_model(model_path)
    img = image.load_img(img_path, target_size=(img_size, img_size))
    arr = image.img_to_array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    preds = model.predict(arr)[0]
    idx = int(np.argmax(preds))
    label = None
    if class_indices:
        inv_map = {v: k for k, v in class_indices.items()}
        label = inv_map.get(idx, str(idx))
    return label, float(preds[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--img-size', type=int, default=224)
    args = parser.parse_args()
    label, prob = predict_image(args.model, args.image, args.img_size)
    print(f'Prediction: {label} ({prob:.4f})')


if __name__ == '__main__':
    main()
