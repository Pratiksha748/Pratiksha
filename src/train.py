import argparse
from src.dataset import create_generators
from src.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os
import json


def main(args):
    os.makedirs(os.path.dirname(args.save_path) or '.', exist_ok=True)
    train_gen, val_gen = create_generators(
        args.train_dir,
        args.val_dir,
        img_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
    )

    model = build_model(num_classes=train_gen.num_classes, img_size=(args.img_size, args.img_size, 3), lr=args.lr)

    callbacks = [
        ModelCheckpoint(args.save_path, save_best_only=True, monitor='val_accuracy', mode='max'),
        EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True),
    ]

    model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, callbacks=callbacks)
    model.save(args.save_path)

    # save class indices for inference mapping
    try:
        class_map = train_gen.class_indices
        class_dir = os.path.dirname(args.save_path) or '.'
        os.makedirs(class_dir, exist_ok=True)
        with open(os.path.join(class_dir, 'class_indices.json'), 'w') as fh:
            json.dump(class_map, fh)
        print('Saved class indices to', os.path.join(class_dir, 'class_indices.json'))
    except Exception as e:
        print('Warning: failed to save class indices:', e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train crop disease classifier')
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--val-dir', required=True)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--save-path', default='models/crop_model.h5')
    args = parser.parse_args()
    main(args)
