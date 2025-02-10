import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix

#import dataset
df_alexa = pd.read_csv('dataset/amazon_alexa_cleaned.csv')


# feature and target 
X = df_alexa.drop(['feedback'], axis=1)
y = df_alexa['feedback']

# split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert boolean columns to integers (0 and 1)
X_train = X_train.astype(int)
X_test = X_test.astype(int)

y_train = np.array(y_train)
y_test = np.array(y_test)

# ANN model
ANN_classifier = tf.keras.models.Sequential()
ANN_classifier.add(tf.keras.layers.Dense(units=400, activation='relu', input_shape=(X_train.shape[1],))) 
ANN_classifier.add(tf.keras.layers.Dense(units=400, activation='relu'))
ANN_classifier.add(tf.keras.layers.Dense(units=1, activation='sigmoid')) #sigmod for binary classification because it gives probability of the class being 1 or 0

# compile model
optimizer_1 = tf.keras.optimizers.Adam(learning_rate=0.0001)

ANN_classifier.compile(optimizer=optimizer_1, loss='binary_crossentropy', metrics=['accuracy']) #binary_crossentropy for binary classification


# train model
history = ANN_classifier.fit(X_train, y_train, batch_size=10, epochs=10)

# evaluate model
y_pred_train = ANN_classifier.predict(X_train)
y_pred_train = (y_pred_train > 0.5)
cm_train = confusion_matrix(y_train, y_pred_train)

sns.heatmap(cm_train, annot=True)
plt.show()

y_pred_test = ANN_classifier.predict(X_test)
y_pred_test = (y_pred_test > 0.5)
cm_test = confusion_matrix(y_test, y_pred_test)

sns.heatmap(cm_test, annot=True)
plt.show()

plt.plot(history.history['loss'])
plt.title('Model Loss Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy Progress During Training')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.legend(['Training Accuracy'])
plt.show()

# Save model
ANN_classifier.save('alexa_feedback_classifier.h5')
ANN_classifier.save('alexa_feedback_classifier.keras')
print('Model saved as alexa_feedback_classifier.h5/.keras')