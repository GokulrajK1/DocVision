import os, argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from effnet import build_efficientnet_b3

DEFAULT_CLASSES = ['chickenpox','measles','acne','eczema','jaundice','healthy']

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', type=str, required=True)
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--out', type=str, default=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'facedx_effnetb3.h5'))
    return ap.parse_args()

def build_datasets(data_dir, img_size, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, label_mode='categorical', validation_split=0.2, subset='training',
        seed=42, image_size=(img_size, img_size), batch_size=batch_size)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, label_mode='categorical', validation_split=0.2, subset='validation',
        seed=42, image_size=(img_size, img_size), batch_size=batch_size)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(AUTOTUNE)
    val_ds = val_ds.prefetch(AUTOTUNE)
    return train_ds, val_ds

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    model = build_efficientnet_b3(num_classes=6, img_size=args.img_size)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_ds, val_ds = build_datasets(args.data_dir, args.img_size, args.batch_size)
    ckpt = ModelCheckpoint(args.out, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt, early, rlrop])
    model.save(args.out)
    print(f"Saved model to {args.out}")

if __name__ == '__main__':
    main()
