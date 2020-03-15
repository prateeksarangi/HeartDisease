from keras.models import Sequential
from keras.layers import Dense
from pandas import read_csv
import matplotlib.pyplot as plt

trainPath = '~/HeartDisease/trainrand.csv'
trainData = read_csv(trainPath)

testPath = "~/HeartDisease/testrand.csv"
testData = read_csv(testPath)

arrayTrain = trainData.values
arrayTest = testData.values

X_train = arrayTrain[:,0:13] 
Y_train = arrayTrain[:,13]

X_test = arrayTest[:, 0:13]
Y_test = arrayTest[:, 13]
model = Sequential()
model.add(Dense(8, input_dim=13, init='uniform', activation='relu'))
model.add(Dense(8, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(8, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(8, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(8, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, Y_train, nb_epoch = 800)

plt.plot(history.history['accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
#plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

scores = model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
