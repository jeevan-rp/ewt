import tensorflow as tf

def build_model(num_classes=10, img_shape=(128,128,3), lr=1e-4):
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
    ])

    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=img_shape, include_top=False, weights='imagenet'
    )
    base_model.trainable = True
    for layer in base_model.layers[:100]:
        layer.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=img_shape),
        data_augmentation,
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['Accuracy']
    )
    return model
