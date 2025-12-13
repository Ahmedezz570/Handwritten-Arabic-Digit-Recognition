import os
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau
)



from dataset import load_and_prepare_data, INPUT_SHAPE, NUM_CLASSES

from model import build_model


BATCH_SIZE = 128
EPOCHS = 30
MODEL_DIR = "results"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.h5")

os.makedirs(MODEL_DIR, exist_ok=True)


def main():
    X_train, X_val, _, y_train, y_val, _, _ = load_and_prepare_data()

    model = build_model(INPUT_SHAPE, NUM_CLASSES)


    model.summary()



    callbacks = [
        ModelCheckpoint(
            MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]



    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(MODEL_DIR, "final_model.h5"))


if __name__ == "__main__":
    main()
