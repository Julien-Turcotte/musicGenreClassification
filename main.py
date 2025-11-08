import librosa.feature
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import EarlyStopping


genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']
X, y = [], []


for g in genres:
    folder = f"genres/{g}"  # adjust path
    for filename in os.listdir(folder):
        song_path = os.path.join(folder, filename)

        y_audio, sr = librosa.load(song_path, duration=30)
        mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        
        max_len = 660
        if mel_db.shape[1] < max_len:
            pad_width = max_len - mel_db.shape[1]
            mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :max_len]

        X.append(mel_db)
        y.append(genres.index(g))

X = np.array(X)
y = to_categorical(np.array(y))


X = X[..., np.newaxis]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# le CNN
model = Sequential([
    Input(shape=(128, 660, 1)),

    # premier axe simple
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    #deuxieme axe plus complexe
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    # flat tout sur un axe pour analyze avancée
    Flatten(),
    Dense(128, activation='relu'),

    # idk gpt overfill or sum
    Dropout(0.3),   

    # output
    Dense(len(genres), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# si t'es tanné d'attendre stop va pas briser le learning
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=1,
    batch_size=32,
    callbacks=[early_stop]
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

model.save('genre_classifier.keras')

plt.plot(history.history['accuracy'], label='Train acc')
plt.plot(history.history['val_accuracy'], label='Val acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

"""
visualisation

X = np.array(X)
print("X shape:", X.shape)

mel_db = X[0]
label_index = y[0]
genre_name = genres[label_index]

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=22050, cmap='magma')
plt.colorbar(format='%+2.0f dB')
plt.title(f"Genre: {genre_name}")
plt.tight_layout()
plt.show()
"""

