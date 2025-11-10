import keras
import os
import librosa
import numpy as np

genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# 1. CHARGER LE MODÈLE
print("Chargement du modèle...")
model = keras.saving.load_model("models/genre_classifier_11-09_20-08-10.keras")
print("✅ Modèle chargé!")


def analyse(song_path):
    """
    processus complet de l'analyse, prediction et affichage pour une chanson
    :param song_path: le full path d'une chanson
    :return: none
    """
    print(f"\nAnalyse de: {os.path.basename(song_path)}")

    # Si chanson > 2 minutes, commence à 60s
    duration = librosa.get_duration(path=song_path)
    offset = 60 if duration > 120 else 0
    y_audio, sr = librosa.load(song_path, duration=30, offset=offset)

    mel = librosa.feature.melspectrogram(y=y_audio, sr=sr, n_mels=128, n_fft=2048, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    max_len = 660
    if mel_db.shape[1] < max_len:
        pad_width = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]


    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_db = mel_db[np.newaxis, ..., np.newaxis]


    prediction = model.predict(mel_db, verbose=0)

    genre_index = np.argmax(prediction)
    genre_name = genres[genre_index]
    confidence = prediction[0][genre_index] * 100

    print("\n" + "="*60)
    print(f"GENRE PRÉDIT: {genre_name.upper()}")
    print(f"CONFIANCE: {confidence:.2f}%")
    print("="*60)

    print("\nProbabilités par genre:")
    for i, genre in enumerate(genres):
        prob = prediction[0][i] * 100
        bar = "█" * int(prob / 2)  # Barre visuelle
        print(f"{genre:12} : {prob:5.2f}% {bar}")
    print("="*60)


# call pour chaque file dans tests/  --WARNING rien mettre autre qu'un suported file
# liste des files qui marche à date (append à mesure des découvertes): .au .mp3 .wav .flac
# not working: .m4a 
for filename in os.listdir("tests"):
    song_path = os.path.join("tests", filename)
    analyse(song_path)
