import pandas as pd
import numpy as np
import string

# Maps alphabets to numbers
di = dict(zip(string.ascii_letters, [ord(c) % 32 for c in string.ascii_letters]))

train_data = []

# Get the data in the dataframe
train_input = pd.read_csv('train_input.csv')

# LSTM inputs a ndarray and for the rows to be of the same column size, we look at the maximum column size
max_length = max(train_input['length'])

for index, row in train_input.iterrows():

    # If length of seq is 100 but the maximum length it can be is 200 then we know we need to add 200-100=100 more columns of zeros
    zeros_to_add = max_length - row['length']

    # For each element in sequence, save the corresponding number so['ABCD'] => ['1.0 2.0 3.0 4.0']
    temp_seq_list = list(row['sequence'])
    seq_list = []
    for i in temp_seq_list:
        seq_list.append(float(di[i]))

    # For each element in q8, save the corresponding number so['ABCD'] => ['1.0 2.0 3.0 4.0']
    temp_q8_list = list(row['sequence'])
    q8_list = []
    for i in temp_q8_list:
        q8_list.append(float(di[i]))

    # Here we multiply by 2(once for sequence and once for q8)
    temp_zero_list = [0] * (2 * zeros_to_add)

    # Just converting every element ot float
    zero_list = [float(i) for i in temp_zero_list]

    # This appends each row as sequence + q8 + number of zeros required
    train_data.append(seq_list + q8_list + temp_zero_list)

train_input = np.array(train_data)

train_output = []
file = np.load('train_output.npz')
for key in file:
    train_output.append(np.average(file[key]))

# Convert the train_output to a numpy matrix
train_output = np.array(train_output)
# Displaying all the shapes
print("Train Input Shape: ",train_input.shape)
print("Train Output Shape: ",train_output.shape)
from tensorflow import keras

# Contructing LSTM having different neurons in different layers, using MSE as the loss function and Adam's optimizer


model = keras.Sequential()
model.add(keras.layers.LSTM(128, input_shape = (1,1382),return_sequences = True))
model.add(keras.layers.Activation('relu'))

model.add(keras.layers.LSTM(64))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(64,activation='relu'))
model.add(keras.layers.Dense(32,activation='relu'))
model.add(keras.layers.Dense(1,activation='relu'))
model.add(keras.layers.Dense(32,activation='relu'))
#

model.compile(loss='mean_squared_error', optimizer='adam')

train_input = train_input.reshape((4554,1,1382))


model.fit(train_input,
                    train_output,
                    epochs=20,
                    validation_split=0.2,
                    verbose=2)
# Constructing the test data the same way we did train data
test_data = []
test_input = pd.read_csv('test_input.csv')
max_length = 691
for index, row in test_input.iterrows():
    zeros_to_add = max_length - row['length']
    temp_seq_list = list(row['sequence'])
    seq_list = []
    for i in temp_seq_list:
        seq_list.append(float(di[i]))

    temp_q8_list = list(row['sequence'])
    q8_list = []
    for i in temp_q8_list:
        q8_list.append(float(di[i]))

    temp_zero_list = [0] * (2 * zeros_to_add)
    zero_list = [float(i) for i in temp_zero_list]
    test_data.append(seq_list + q8_list + temp_zero_list)

test_input = np.array(test_data)
test_input = test_input.reshape((224,1,1382))
p = model.predict(test_input)
final_matrix = []
test_input = pd.read_csv('test_input.csv')

for index, row in test_input.iterrows():
    # For each test example, create a temp_matrix having all values same as predicted value
    temp_matrix = np.full((row['length'], row['length']), p[index][0])

    # Set the diagonal values to 0
    np.fill_diagonal(temp_matrix, 0)

    # Append to final_matrix
    final_matrix.append(temp_matrix)

np.savez('lstm_baseline.npz', *final_matrix)


import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(0,116,117)
y=np.random.normal(x)    # add some noise

#plt.plot(x,y,'r.') # x vs y
plt.plot(x,x,'k-') # identity line

plt.xlim(0,116)
plt.ylim(116,0)
ax = plt.gca()

ax.set_facecolor('xkcd:salmon')
ax.set_facecolor((1.0, 0.47, 0.42))
plt.show()


