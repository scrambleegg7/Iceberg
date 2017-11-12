
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Activation, ELU, Add
from keras.layers import GlobalMaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping

from keras.preprocessing.image import ImageDataGenerator


def get_model(p_activation="elu", k_size = (5,5), dropout=0.25):
    bn_model = 0.99
    #p_activation = "elu"
    input_1 = Input(shape=(75, 75, 3), name="X_1")
    input_2 = Input(shape=[1], name="angle")

    img_1 = Conv2D(32, kernel_size = k_size, activation=p_activation, padding="same") ((BatchNormalization(momentum=bn_model))(input_1))
    img_1 = Conv2D(32, kernel_size = k_size, activation=p_activation, padding="same") (img_1)
    img_1 = MaxPooling2D((2,2)) (img_1)
    img_1 = Dropout(dropout)(img_1)

    img_1 = Conv2D(64, kernel_size = k_size, activation=p_activation, padding="same")((BatchNormalization(momentum=bn_model))(img_1))
    img_1 = Conv2D(64, kernel_size = k_size, activation=p_activation, padding="same")(img_1)
    img_1 = MaxPooling2D(pool_size=(2, 2))(img_1)
    img_1 = Dropout(dropout)(img_1)

    print("img1 shape", img_1.shape)

    img_resid = Conv2D(128, kernel_size = k_size, activation=p_activation, padding="same")((BatchNormalization(momentum=bn_model))(img_1))
    img_resid = Conv2D(128, kernel_size = k_size, activation=p_activation, padding="same")(img_resid)
    img_resid = Dropout(dropout)(img_resid)
    print("img_resid shape1", img_resid.shape)

    img_resid = Conv2D(64, kernel_size = k_size, activation=p_activation, padding="same")((BatchNormalization(momentum=bn_model))(img_resid))
    img_resid = Conv2D(64, kernel_size = k_size, activation=p_activation, padding="same")(img_resid)
    print("img_resid shape2", img_resid.shape)

    cnn_resid_added = Add()([img_1, img_resid])
    print("cnn_resid shape", cnn_resid_added.shape)

    top_cnn = Conv2D(128, kernel_size = k_size, activation=p_activation, padding="same")((BatchNormalization(momentum=bn_model))(cnn_resid_added))
    top_cnn = Conv2D(128, kernel_size = k_size, activation=p_activation, padding="same")(top_cnn)
    top_cnn = MaxPooling2D((2,2)) (top_cnn)
    top_cnn = Conv2D(256, kernel_size = k_size, activation=p_activation, padding="same")((BatchNormalization(momentum=bn_model))(top_cnn))
    top_cnn = Conv2D(256, kernel_size = k_size, activation=p_activation, padding="same")(top_cnn)
    top_cnn = Dropout(0.25)(top_cnn)
    top_cnn = MaxPooling2D((2,2)) (top_cnn)
    top_cnn = Conv2D(512, kernel_size = k_size, activation=p_activation, padding="same")((BatchNormalization(momentum=bn_model))(top_cnn))
    top_cnn = Conv2D(512, kernel_size = k_size, activation=p_activation, padding="same")(top_cnn)
    top_cnn = Dropout(dropout)(top_cnn)
    top_cnn = MaxPooling2D((2,2)) (top_cnn)

    top_cnn = GlobalMaxPooling2D()(top_cnn)
    print("top_cnn shape", top_cnn.shape)

    dense_ayer = ELU()(BatchNormalization(momentum=bn_model)( Dense(512, activation=None)(top_cnn) ))
    dense_ayer = Dropout(0.5)(dense_ayer)
    dense_ayer = ELU()(BatchNormalization(momentum=bn_model)( Dense(256, activation=None)(dense_ayer) ))
    dense_ayer = Dropout(0.5)(dense_ayer)

    # 2 for One Hot code for Binary
    #output = Dense(2, activation="softmax")(dense_ayer)
    output = Dense(1, activation="softmax")(dense_ayer)

    model = Model([input_1],  output)
    #optimizer = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    return model
