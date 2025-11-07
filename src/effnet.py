from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, models

def build_efficientnet_b3(num_classes: int = 6, img_size: int = 224) -> models.Model:
    base = EfficientNetB3(
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3),
        pooling='avg'
    )
    x = layers.Dropout(0.3)(base.output)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs=base.input, outputs=out)
