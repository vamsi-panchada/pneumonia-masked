import streamlit as st
from tensorflow.keras import layers, Model
import getData
import cv2
import numpy as np
import pickle


channel_axis = -1


@st.cache_resource
def unet():
    input_size=(512, 512, 1)
    inputs = layers.Input(input_size)
    
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    up6 = layers.concatenate([layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = layers.concatenate([layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = layers.concatenate([layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = layers.concatenate([layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = layers.Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    pmodel = Model(inputs=[inputs], outputs=[conv10])

    try:
        pmodel.load_weights('MaskWeights.hdf5')
    except:
        file_id = '1S9NafzhX8kb6NqBp2PiJx1mx5EHpLlwN'
        destination = 'MaskWeights.hdf5'
        getData.download_file_from_google_drive(file_id, destination)
        pmodel.load_weights('MaskWeights.hdf5')

    return pmodel


@st.cache_resource
def Pmodel():
    img_input = layers.Input(shape = (224, 224, 1))
    x = layers.Conv2D(32, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block1_conv1')(img_input)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block1_bn1')(x)
    x = layers.Activation('relu', name = 'block1_act1')(x)
    x = layers.Conv2D(32, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block1_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block1_bn2')(x)
    x = layers.Activation('relu', name = 'block1_act2')(x)
    x = layers.MaxPooling2D((2, 2),
                            strides=(2, 2),
                            padding='same',
                            name='block1_pool')(x)

    # block 2
    x = layers.Conv2D(64, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block2_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block2_bn1')(x)
    x = layers.Activation('relu', name = 'block2_act1')(x)
    x = layers.Conv2D(64, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block2_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block2_bn2')(x)
    x = layers.Activation('relu', name = 'block2_act2')(x)
    x = layers.MaxPooling2D((2, 2),
                            strides=(2, 2),
                            padding='same',
                            name='block2_pool')(x)

    # block 3
    x = layers.Conv2D(128, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block3_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block3_bn1')(x)
    x = layers.Activation('relu', name = 'block3_act1')(x)
    x = layers.Conv2D(128, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block3_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block3_bn2')(x)
    x = layers.Activation('relu', name = 'block311_act2')(x)
    x = layers.MaxPooling2D((3, 3),
                            strides=(3, 3),
                            padding='same',
                            name='block3_pool')(x)

    x = layers.Conv2D(256, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block31_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block31_bn1')(x)
    x = layers.Activation('relu', name = 'block31_act1')(x)
    x = layers.Conv2D(128, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block31_conv2')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block31_bn2')(x)
    x = layers.Activation('relu', name = 'block31_act2')(x)
    x = layers.MaxPooling2D((3, 3),
                            strides=(3, 3),
                            padding='same',
                            name='block31_pool')(x)

  # block 4
    x = layers.Conv2D(1024, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block41_conv1')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block41_bn1')(x)
    x = layers.Activation('relu', name = 'block41_act1')(x)
    x = layers.Conv2D(512, (3,3),
                      padding = 'same', use_bias = False,
                      name = 'block41_conv2')(x)
    x = layers.Dropout(0.5, name = 'block4_dropout')(x)
    x = layers.BatchNormalization(axis = channel_axis, name = 'block4_bn2')(x)
    x = layers.Activation('relu', name = 'block4_act2')(x)
    x = layers.MaxPooling2D((3, 3),
                            strides=(3, 3),
                            padding='same',
                            name='block4_pool')(x)
    x = layers.Flatten(name='flatten')(x)
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.Dense(1024, activation='relu', name='fc11')(x)
    x = layers.Dense(512, activation='relu', name='fc3')(x)
    x = layers.Dense(512, activation='relu', name='fc4')(x)
    x = layers.Dense(256, activation='relu', name='fc5')(x)
    x = layers.Dense(64, activation='relu', name='fc6')(x)
    x = layers.Dense(2, activation='softmax', name='predictions')(x)
    model = Model(inputs=img_input, outputs=x, name = 'own_build_model')

    try:
        model.load_weights('PneumoniaWeights.hdf5')
    except:
        file_id = '1h6Jfq93KF59-N48t9HfY02_yUTAJdBCU'
        destination = 'PneumoniaWeights.hdf5'
        getData.download_file_from_google_drive(file_id, destination)
        model.load_weights('PneumoniaWeights.hdf5')

    return model


model = unet()
pmodel = Pmodel()
knn_model = pickle.load(open('knn_model.pkl', 'rb'))

st.title('Pneumonia Detection Application using Segmentation')
st.text('Please Upload a Chest X-RAY Image to detect the Pneumonia.')

uploadedFiles = st.file_uploader('Upload Chest X-Rays', accept_multiple_files=True)

imageArray = []
betaColumnArray = []
midColumnArray = []

maskCheck = st.checkbox('Display Masks')
# maskRadio = False
if maskCheck:
    maskRadio = st.radio('which you want to display', ['mask', 'segmented'])

if uploadedFiles:
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    for upload_file in uploadedFiles:

        file_bytes = np.asarray(bytearray(upload_file.read()), dtype=np.uint8)
        im = cv2.imdecode(file_bytes, 1)
        col1, mid, col2 = st.columns([20, 20, 20])
        col1.image(cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC), channels="BGR", caption=upload_file.name)
        # col1.write(upload_file.name)
        imageArray.append(im)
        betaColumnArray.append(col2)
        midColumnArray.append(mid)
        st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)

if len(imageArray)>0:
    if st.button('Predict'):
        for im, mid, col2 in zip(imageArray, midColumnArray, betaColumnArray):
            im = cv2.resize(im, (512, 512))[:,:,0]
            im = im.reshape(1, 512, 512, 1)
            mask = model.predict(im).reshape(512, 512)
            im = im.reshape(512, 512)
            im[mask==0]=0
            if maskCheck:
                if maskRadio == 'mask':
                    mid.image(mask, channels='GRAY', caption='Generated Mask')
                elif maskRadio == 'segmented':
                    mid.image(im, channels='GRAY', caption='Segmented Image')
            else:
                # print('else column')
                mid.header('Prediction ==>')
            im = cv2.resize(im, (224, 224), interpolation=cv2.INTER_CUBIC).reshape(1, 224, 224, 1)
            im = im.astype(np.float32)/255.
            
            classes = pmodel.predict(im)

            res = knn_model.predict(classes)

            if res == 0:
                col2.title(':green[NORMAL]\nYou are fine No need to Worry. 😊')
                col2.write('with probability of '+str(classes[0][int(res)]*100)+'%')
            else:
                col2.title(':red[PNEUMONIA IS FOUND]\nGet Well Soon ✌🏻')
                # col2.write('with probability of '+str(classes[int(res)]*100)+'%')
                col2.write('with probability of '+str(classes[0][int(res)]*100)+'%')

            # if np.argmax(classes)==0:
            #     col2.title(':green[NORMAL]\nYou are fine No need to Worry. 😊')
                
            # else:
            #     col2.title(':red[PNEUMONIA IS FOUND]\nGet Well Soon ✌🏻')

imageArray.clear()
betaColumnArray.clear()