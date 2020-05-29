import tensorflow as tf
import numpy as np
import os
import datetime
import pandas as pd
import matplotlib.pyplot as plt
def show_plot(trials):
  for i in trials:
    plt.plot(list(range(-HISTORY_SIZE, 0)),x_val[i, :, 0], label="History")
    plt.plot(y_val[i], "bo", label="True Future")
    a=x_val[i]
    a=a.reshape(1, HISTORY_SIZE, 5)
    plt.plot(model.predict(a, batch_size=1), "ro", label="Predicted Future")
    plt.legend(loc="upper left")
    plt.show()
def define_model():
  model=tf.keras.models.Sequential()
  model.add(tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(x_train.shape[-2:])))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.LSTM(50, return_sequences=True))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.LSTM(50, return_sequences=True))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.LSTM(50))
  model.add(tf.keras.layers.Dropout(0.2))
  model.add(tf.keras.layers.Dense(1))
  model.compile(optimizer='adam',loss='mean_squared_error')
  return model
def str_to_date(date_str):
  year, month, day=date_str.split('-')
  return datetime.date(int(year), int(month), int(day))
def plot_train_history(history, title):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()

    plt.show()
def multivariate_data(dataset, target, history_size, target_size, step):
  x=[]
  y=[]

  start_index=history_size
  end_index=dataset.shape[0]-target_size
  for i in range(start_index, end_index):
    indicies=range(i-history_size, i, step)
    x.append(dataset[indicies])
    y.append(target[i+target_size])
  x=np.array(x)
  y=np.array(y)
  return x, y
STOCK_SYMBOL='AAPL'
path_to_file=('/Users/johnathanxie/Documents/Python/datasets/AAPL_Stock.csv')
HISTORY_SIZE=100
dataset=pd.read_csv(path_to_file, usecols=range(1,6), dtype='float32')
dataset=np.array(dataset)
dataset=dataset[4000:9000]
BATCH_SIZE=50
for i in range(dataset.shape[1]):
  dataset[:, i]=(dataset[:, i]-dataset[:, i].mean())/dataset[:, i].std()
x_train, y_train=multivariate_data(dataset[:4200], dataset[:4200, 0], HISTORY_SIZE, 1, 1)
print(x_train.shape)
print(y_train.shape)
x_val, y_val=multivariate_data(dataset[4200:], dataset[4200:, 0], HISTORY_SIZE, 1, 1)
print(x_val.shape)
print(y_val.shape)
print(x_train.shape[-2:])
train_data=tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data=train_data.cache().shuffle(x_train.shape[0]).batch(BATCH_SIZE, drop_remainder=True).repeat()

val_data=tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_data=train_data.batch(BATCH_SIZE, drop_remainder=True).repeat()

model=tf.keras.models.load_model('./AAPL_basic')
print(model.summary())
#checkpoint_dir='./stonks_training_checkpoints'
#checkpoint_prefix=os.path.join(checkpoint_dir, "ckpt_{epoch}")
#checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
#model.save('./AAPL_basic')
#history=model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val), callbacks=[checkpoint_callback],
#                  steps_per_epoch=75, validation_steps=25)
#plot_train_history(history, "Train_History")
show_plot(range(-200, -100, 5))
