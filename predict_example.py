"An example of predicting a music genre from a custom audio file"
import librosa
import logging
import numpy as np

from keras.models import model_from_json
from matplotlib import pyplot as plt
from numpy import floor
from sklearn import metrics

from GenerateSpectrogramData import GenerateSpectrogramData

# set logging level
logging.getLogger("tensorflow").setLevel(logging.ERROR)

def load_model(model_path, weights_path):
    with open(model_path, "r") as model_file:
        trained_model = model_from_json(model_file.read())
    trained_model.load_weights(weights_path)
    trained_model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    return trained_model


def extract_audio_features(file):
    SR = 12000
    N_FFT = 512
    N_MELS = 96
    HOP_LEN = 256
    DURA = 29.12
    DURA_TRASH = 0

    src, sr = librosa.load(file)
    n_sample = src.shape[0]
    n_sample_fit = int(DURA * SR)
    n_sample_trash = int(DURA_TRASH * SR)

    # trim tail and head
    src = src[n_sample_trash:(n_sample - n_sample_trash)]
    n_sample = n_sample - 2 * n_sample_trash

    ret = np.zeros((96, 1366), dtype=np.float32)

    if n_sample < n_sample_fit:  # if too short
        src = np.hstack((src, np.zeros((int(DURA * SR) - n_sample,))))
        logam = librosa.amplitude_to_db
        melgram = librosa.feature.melspectrogram
        ret = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                            n_fft=N_FFT, n_mels=N_MELS) ** 2, )
    elif n_sample > n_sample_fit:  # if too long
        N = int(floor(n_sample / n_sample_fit))
        src_total = src
        for i in range(0, N):
            src = src_total[(i * n_sample_fit):(i + 1) * n_sample_fit]

            logam = librosa.amplitude_to_db
            melgram = librosa.feature.melspectrogram
            retI = logam(melgram(y=src, sr=SR, hop_length=HOP_LEN,
                                 n_fft=N_FFT, n_mels=N_MELS) ** 2, )
            ret = np.concatenate((ret, retI), axis=0)

    return ret


def get_genre(model, music_path):
    "Predict genre of music using a trained model"
    feature = extract_audio_features(music_path)
    shape0 = feature.shape[0]
    frames_num = int(shape0 / 192)
    trashes = shape0 - frames_num * 192
    feature = feature[int(trashes / 2): shape0 - int(trashes / 2), :]
    feature = feature.reshape((frames_num, 1, 192, 1366))
    prediction = model.predict(feature)
    prediction = np.mean(prediction, axis=0)
    return prediction


if __name__ == "__main__":
    CLASSICAL_PATH = "test/classical.au"
    HIPHOP_PATH = "test/hiphop.au"
    JAZZ_PATH = "test/jazz.au"
    METAL_PATH = "test/metal.mp3"
    POP_PATH = "test/pop.au"
    REGGAE_PATH = "test/reggae.au"
    MODEL = load_model("cnn_genre_classifier_lstm_25_500_sgd_80w.json",
                       "cnn_genre_classifier_lstm_25_500_sgd_80w.h5")
    CLASSICAL_PREDICTION = get_genre(MODEL, CLASSICAL_PATH)
    HIPHOP_PREDICTION = get_genre(MODEL, HIPHOP_PATH)
    JAZZ_PREDICTION = get_genre(MODEL, JAZZ_PATH)
    METAL_PREDICTION = get_genre(MODEL, METAL_PATH)
    POP_PREDICTION = get_genre(MODEL, POP_PATH)
    REGGAE_PREDICTION = get_genre(MODEL, REGGAE_PATH)
    # print confusion matrices 6 x 6
    predicted = CLASSICAL_PREDICTION, HIPHOP_PREDICTION, JAZZ_PREDICTION, METAL_PREDICTION, POP_PREDICTION, REGGAE_PREDICTION
    fig, ax = plt.subplots()
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=np.array(predicted), display_labels=GenerateSpectrogramData.genre_list)
    cm_display.plot()
    plt.show()