from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_generators(train_dir, val_dir, img_size=(224, 224), batch_size=32):
    """Create Keras ImageDataGenerator train and validation generators.

    Expects directory structure:
      train_dir/class_x/xxx.png
      train_dir/class_y/yyy.png
    """
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_gen = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
    )

    val_gen = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
    )

    return train_gen, val_gen
