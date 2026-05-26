from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models, optimizers

def build_model(num_classes, img_size=(224, 224, 3), lr=1e-4):
    base = MobileNetV2(include_top=False, input_shape=img_size, weights='imagenet')
    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model
