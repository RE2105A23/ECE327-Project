import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.model_selection import train_test_split


# Step 1: Load and Preprocess the Data
# df = pd.read_csv('/Users/sjsb/git/ece327/datasets/emotion_model/')
df = pd.read_csv('/Users/sjsb/git/ece327/datasets/icml_face_data.csv/icml_face_data.csv')
# Check the first few rows of the DataFrame to see if 'pixels' column exists
#print(df.head())
# Check the column names of the DataFrame
#print(df.columns)
df.columns = df.columns.str.strip()
X = df['pixels'].apply(lambda x: [int(pixel) for pixel in x.split()])
X = np.array(X.tolist()).reshape(-1, 48, 48, 1).astype('float32') / 255.0  # Normalize pixel values
y = to_categorical(df['emotion'])

# Step 2: Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Design the Model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
    Flatten(),
    Dense(7, activation='softmax')
])

# Step 4: Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the Model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

# Step 6: Save the Trained Model
# model.save('/Users/sjsb/git/ece327/datasets/emotion_model.h5')
model.save('/Users/sjsb/git/ece327/datasets/emotion_model')  # This will save in the TensorFlow SavedModel format by default