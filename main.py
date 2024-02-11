import os.path
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tp

mnits = tp.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnits.load_data()
X_train = tp.keras.utils.normalize(X_train, axis=1)
X_test = tp.keras.utils.normalize(X_test, axis=1)

# model = tp.keras.models.Sequential()
# model.add(tp.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tp.keras.layers.Dense(128, activation="relu"))
# model.add(tp.keras.layers.Dense(128, activation="relu"))
# model.add(tp.keras.layers.Dense(10, activation="softmax"))
#
# model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
# model.fit(X_train, y_train, epochs=10)
# model.save('mnist_model')

model = tp.keras.models.load_model('mnist_model')

PIC_NUM = 1
while os.path.isfile(f"pred/pic{PIC_NUM}.png"):
    try:
        pic = cv2.imread(f"pred/pic{PIC_NUM}.png")[:, :, 0]
        pic = np.invert(np.array([pic]))
        prediction = model.predict(pic)
        print(f"pic{PIC_NUM} is: {np.argmax(prediction)}")
        plt.imshow(pic[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally:
        PIC_NUM += 1