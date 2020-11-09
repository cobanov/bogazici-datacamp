import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

#Kullanılacak parametreleri ileride değiştirmeyi kolaylaştırmak amacıyla tanımlıyoruz.
batch_size = 128
num_classes = 10
epochs = 5

# MNIST veri setini indiriyoruz.
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#Inputlarımı oluşturacağımız modelleri için düzenliyoruz.
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

#X verisini inceleyelim
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples') 

#Etiketlerimizi Gerekli Şekline Getiriyoruz
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#Y Verisini İnceleyelim
print(y_train.shape, 'train samples')
print(y_test.shape, 'test samples') 

#Modeli Oluşturalım

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,))) #Giriş Katmanı - Çıkış Değeri 512, Giriş Sayısı X verisiyle eşleniyor.
model.add(Dropout(0.2))                                      # Overfitting problemini çözmek amacıyla sinir ağına dropout ekliyoruz.
model.add(Dense(512, activation='relu'))                     #Hidden layer olarak yine çıkışı 512 olan bir katman ekliyoruz.
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))          #Çıkış katmanımız toplamda nöron sayısını etiketlerin sınıfları kadar yapıyoruz, bu sayede aktif olan nöron bize sayı değerini verecek.

model.summary()                                              #Ağın genel özetine bakalım.

model.compile(loss='categorical_crossentropy',               #Modeli derliyoruz, bu bölümde loss fonksiyonumu, optimizasyonu ve ölçüm değerinin türünü belirliyoruz.
              optimizer=Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,                        #Modeli çalıştıralım, verileri verip ilgili değerleri çalıştırıyoruz.
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)            #Hesaplamaları çıktı olarak almak için hesaplayalım.

print('Test loss:', score[0])
print('Test accuracy:', score[1])