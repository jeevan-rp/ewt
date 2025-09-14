import tensorflow as tf

def train_model(model, datatrain, datavalid, epochs=15):
    early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True
    )
    history = model.fit(datatrain, validation_data=datavalid, epochs=epochs, callbacks=[early])
    return history
