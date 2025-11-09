import librosa.feature
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
import datetime



genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

"""
conventions du ML X = input, y = output
X est l'array qui represente les chansons
y est le genre prédit
"""
X, y = [], []

for g in genres:
    folder = f"genres/{g}"
    for filename in os.listdir(folder):
        song_path = os.path.join(folder, filename)

        y_audio, sr = librosa.load(song_path, duration=30)
        # creation spectrogramme
        mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
        # Convertie en decibel
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # pour consitent data
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

# Normaliser les données entre 0 et 1 (important!)
X = (X - X.min()) / (X.max() - X.min())

X = X[..., np.newaxis] # ajout d'un 4e axe
# 20% des data va dans test le reste on train avec
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42)


# le CNN
model = Sequential([
    Input(shape=(128, 660, 1)),

    # premier axe simple
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    #deuxieme axe plus complexe
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Troisième couche pour plus de profondeur
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # flat tout sur un axe pour analyze avancée
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    # output
    Dense(len(genres), activation='softmax') # softmax = trouve le genre le plus probable
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Nombre d'échantillons:", len(X_train), "train,", len(X_test), "test")
print("Shape X_train:", X_train.shape)
print("Shape y_train:", y_train.shape)

# stop quand le model n'apprend plus
# patience = nombre d'epochs à attendre sans amélioration avant d'arrêter
early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20, # à 30 l'accuracy baisse
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print("Test accuracy:", test_acc)

model.save(f'genre_classifier_{datetime.datetime.now().strftime("%m-%d_%H-%M-%S")}.keras')


# graph
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

