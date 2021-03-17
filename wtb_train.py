import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

gpu = tf.config.list_physical_devices('GPU')
try:
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
	pass

# CURR_DIR = "WTB_Images_03152021"
CURR_DIR = "/home/boyuan/custom_data/pposs/train"
CLASSES = ("bear", "coyote", "deer", "other", "empty")
MODEL_DIR = "./checkpoints/imagenet.h5"

# Set PRETRAIN_MODEL true if we want to train a new model from scratch
PRETRAIN_MODEL = False
IMAGE_SIZE = (1920, 1080)
INPUT_SHAPE = IMAGE_SIZE + (3,)
BATCH_SIZE = 16
NUM_CLASS = 5
EPOCHS = 3 

def build_model(input_shape=(1920, 1080, 3), num_class=5):
    input_t = keras.Input(shape=input_shape)
    res_model = keras.applications.ResNet50(include_top=False, 
                                            weights="imagenet", 
                                            input_tensor=input_t)

    # Lock up all layers of ResNet50 except the last block
    for layer in res_model.layers[:143]:
        layer.trainable = False
        
    model = keras.models.Sequential()
    model.add(res_model)
    model.add(keras.layers.Flatten())
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(256, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(128, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.BatchNormalization())
    # model.add(keras.layers.Dense(64, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(5, activation='softmax'))
    
    return model


if __name__ == "__main__":

    num_skipped = 0
    for folder_name in CLASSES:
        folder_path = os.path.join(CURR_DIR, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        CURR_DIR,
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        CURR_DIR,
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode='categorical',
    )

    train_ds = train_ds.prefetch(buffer_size=BATCH_SIZE)
    val_ds = val_ds.prefetch(buffer_size=BATCH_SIZE)
    

    # load trained model
    if not PRETRAIN_MODEL:
        model = load_model(MODEL_DIR)
    else:
        model = build_model(INPUT_SHAPE, NUM_CLASS)

    # print (model.summary())

    callbacks = [keras.callbacks.ModelCheckpoint(filepath = "save_at_{epoch}.h5",
                                                 monitor = "val_acc",
                                                 mode = "max",
                                                 save_best_only = True),]

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"],)

    model.fit(train_ds, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE, 
              callbacks=callbacks, 
              validation_data=val_ds,
              verbose=1,)
    
    model.save("./checkpoints/imagenet.h5")
