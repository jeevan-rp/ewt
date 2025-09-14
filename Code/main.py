from data_preprocessing import load_datasets, plot_sample_images
from model import build_model
from train import train_model
from evaluate import plot_training_history, evaluate_model

trainpath = r"C:\Users\vindh\OneDrive\Desktop\EWT\EWT dataset\modified-dataset/train"
validpath = r"C:\Users\vindh\OneDrive\Desktop\EWT\EWT dataset\modified-dataset/val"
testpath  = r"C:\Users\vindh\OneDrive\Desktop\EWT\EWT dataset\modified-dataset/test"

datatrain, datavalid, datatest = load_datasets(trainpath, validpath, testpath)
class_names = datatrain.class_names
plot_sample_images(datatrain, class_names)

model = build_model(num_classes=len(class_names))
history = train_model(model, datatrain, datavalid, epochs=15)

plot_training_history(history)
evaluate_model(model, datatest, class_names)

model.save("Efficient_classify.keras")
