import tensorflow as tf
import matplotlib.pyplot as plt

def load_datasets(trainpath, validpath, testpath, img_size=(128,128), batch_size=32):
    datatrain = tf.keras.utils.image_dataset_from_directory(
        trainpath, shuffle=True, image_size=img_size, batch_size=batch_size
    )
    datavalid = tf.keras.utils.image_dataset_from_directory(
        validpath, shuffle=True, image_size=img_size, batch_size=batch_size
    )
    datatest = tf.keras.utils.image_dataset_from_directory(
        testpath, shuffle=False, image_size=img_size, batch_size=batch_size
    )
    return datatrain, datavalid, datatest

def plot_sample_images(dataset, class_names):
    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(12):
            ax = plt.subplot(4, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.show()
