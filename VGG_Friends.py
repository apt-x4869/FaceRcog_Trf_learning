from keras.applications import VGG16

# VGG16 was designed to work on 224 x 224 pixel input images sizes
img_rows = 224
img_cols = 224 

# Loads the VGG16 model with No Top
model = VGG16(weights = 'imagenet', 
              include_top = False, 
              input_shape = (img_rows, img_cols, 3))

# Let's print our layers 
for (i,layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__ + "\t" ,layer.trainable)

# Freeze all layers
for layer in model.layers:
    layer.trainable = False
# Let's print our layers 
for (i,layer) in enumerate(model.layers):
    print(str(i) + " " + layer.__class__.__name__ + "\t" ,layer.trainable)

# For adding top and bootom layers
def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model

num_classes = 16
FC_Head = addTopModel(model, num_classes)
modelnew = Model(inputs=model.input, outputs=FC_Head)

print(modelnew.summary())

from keras.preprocessing.image import ImageDataGenerator
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory="Training/",target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory="Validation/", target_size=(224,224))

from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("friends_vgg.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint]

# Note we use a very small learning rate 
modelnew.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])

nb_train_samples = 3777
nb_validation_samples = 546
epochs = 6
batch_size = 16

history = modelnew.fit_generator(
    traindata,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = testdata,
    validation_steps = nb_validation_samples // batch_size)

modelnew.save("friends_vgg16.h5")
