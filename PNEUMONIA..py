import tensorflow as tf
from tensorflow.keras import layers,models
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os
import cv2
import zipfile

zip_path=r"C:\Users\ABCD\Downloads\archive (4).zip"
extract_path="chest_xray_data"
with zipfile.ZipFile(zip_path,"r") as zip_ref:
    zip_ref.extractall(extract_path)

print("Extraction done!!!!")

img_size=(227,227) #BASED ON ALEX NET ARCHITECTURE
batch_size=32

train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen=ImageDataGenerator(rescale=1./255)

train_dir=os.path.join(extract_path,"chest_xray","train")
test_dir=os.path.join(extract_path,"chest_xray","test")

train_generator=train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

test_generator=test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode="binary"
)

model=models.Sequential([
    layers.Conv2D(96,kernel_size=11,strides=4,activation="relu",input_shape=(227,227,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=3,strides=2),

    layers.Conv2D(256,kernel_size=5,strides=1,padding="same",activation="relu"),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=3,strides=2),

    layers.Conv2D(384,kernel_size=3,strides=1,padding="same",activation="relu"),
    layers.Conv2D(384,kernel_size=3,padding="same",activation="relu"),
    layers.Conv2D(256,kernel_size=3,padding="same",activation="relu"),
    layers.MaxPooling2D(pool_size=3,strides=2),

    layers.Flatten(),
    layers.Dense(units=4096,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(units=4096,activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(units=1,activation="sigmoid")
])

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()

history=model.fit(
    train_generator,validation_data=test_generator,epochs=15
)

#TRAINING AND VALIDATION ACCURACY
plt.plot(history.history["accuracy"],label="train-accuracy")
plt.plot(history.history["val_accuracy"],label="validation-accuracy")
plt.title("Model Accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.legend()
plt.grid(True)
plt.show()

#TRAINING AND VALIDATION LOSS
plt.plot(history.history["loss"],label="train-loss")
plt.plot(history.history["val_loss"],label="validation-loss")
plt.plot("Model Loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.grid(True)
plt.show()

model.save("pneumonia_model")

model=load_model("pneumonia_model")

img_path=input("Enter path for chest X-ray image:")

img=image.load_img(img_path,target_size=(227,227))
img_array=image.img_to_array(img)/255.0
img_array = np.expand_dims(img_array, axis=0)

pred = model.predict(img_array)[0][0]
result = "Pneumonia Detected (PLEASE TAKE AN APPOINTMENT WITH THE DOC)" if pred > 0.5 else "Normal (YOU ARE GOOD TO GO!!!!!) "

plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {result}")
plt.show()

def generate_gradcam(img_path, model, last_conv_layer_name="conv2d_5"):
       
    img = image.load_img(img_path, target_size=(227, 227))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    pred_index = int(preds[0][0] > 0.5)
   
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

  
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, pred_index]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
   
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
   
    img = cv2.imread(img_path)
    img = cv2.resize(img, (227, 227))
    heatmap = cv2.resize(heatmap.numpy(), (227, 227))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
   
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(f"Prediction: {'Pneumonia' if pred_index else 'Normal'}")
    plt.show()

generate_gradcam(img_path, model)
