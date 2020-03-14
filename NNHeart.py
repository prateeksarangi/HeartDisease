from keras.models import Sequential
from keras.layers import Dense, Conv2D
from pandas import read_csv

filename = 'heart.csv'
dataframe = read_csv(filename)

array = dataframe.values

X = array[:,0:13] 
Y = array[:,13]

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(303, 13), activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dense(12, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(12, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(8, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(8, init='uniform', activation='sigmoid', use_bias=True))
model.add(Dense(1, init='uniform', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, Y, nb_epoch = 800)

scores = model.evaluate(X, Y)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))