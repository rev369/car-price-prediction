import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as py
import numpy as np

data=pd.read_csv("/train.csv")
print(data.shape)
sns.pairplot(data[['years','km','rating','condition','economy','top speed','hp','torque','current price']],diag_kind='kde')

tensor_data=tf.cast(tf.constant(data),dtype=tf.float32)
tensor_data=tf.random.shuffle(tensor_data)

x_tensor=tensor_data[:,3:-1]
print(x_tensor.shape)

y_tensor=tensor_data[:,-1]
y=tf.expand_dims(y_tensor,axis=1)
print(y.shape)

test=0.1
train=0.8
validation=0.1
dataset=len(x_tensor)

x_train=x_tensor[:int(dataset*train)]
y_train=y[:int(dataset*train)]


x_test=x_tensor[:int(dataset*test)]
y_test=y[:int(dataset*test)]


x_val=x_tensor[int(dataset*train):int(dataset+validation)]
y_val=y[int(dataset*train):int(dataset+validation)]

nor=tf.keras.layers.Normalization()
nor.adapt(x_train)
nor(x_train)

model=tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=(8,)),nor,tf.keras.layers.Dense(256,activation="relu"),tf.keras.layers.Dense(256,activation="relu"),tf.keras.layers.Dense(128,activation="relu"),tf.keras.layers.Dense(1)])

model.summary()
tf.keras.utils.plot_model(model,to_file='model.png',show_shapes=True)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2),loss=tf.keras.losses.MeanAbsoluteError())

his=model.fit(x_train,y_train,epochs=100,verbose=1,validation_data=(x_val,y_val))



py.plot(his.history["loss"])
py.plot(his.history["val_loss"])
py.xlabel("epochs")
py.ylabel("loss")
py.legend(["train","val"])
py.show()


model.evaluate(x_test,y_test)
y_pred=list(model.predict((x_test))[:,0])
print(len(y_pred))
y_true=list(y_test[:,0])


ind=np.arange(100)
print(ind)
width=0.4
y_true=list(y_test[:,0])
py.bar(ind,y_pred,width,label="pred price")
py.bar(ind+width,y_true,width,label="actual price")
py.show()
