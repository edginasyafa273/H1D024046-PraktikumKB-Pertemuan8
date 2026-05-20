# H1D024046-PraktikumKB-Pertemuan8
# Implementasi Convolutional Neural Network (CNN) untuk Klasifikasi Gambar Rock-Paper-Scissors

Program ini merupakan implementasi metode **Convolutional Neural Network (CNN)** menggunakan **TensorFlow** dan **Keras** untuk mengklasifikasikan gambar ke dalam tiga kategori, yaitu **rock (batu)**, **paper (kertas)**, dan **scissors (gunting)**. Dataset diproses menggunakan **ImageDataGenerator**, kemudian model CNN dilatih dan dievaluasi untuk melihat performanya.

---

## 1. Library yang Digunakan

Program menggunakan beberapa library Python berikut:

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
```

Penjelasan library:

**NumPy**

Digunakan untuk membantu operasi numerik dan manipulasi array.

**Pandas**

Digunakan untuk pengolahan data pendukung.

**TensorFlow dan Keras**

Digunakan untuk membangun, melatih, dan mengevaluasi model CNN.

**ImageDataGenerator**

Digunakan untuk preprocessing data gambar serta membagi data training dan validation.

**zipfile**

Digunakan untuk mengekstrak dataset dari file ZIP.

---

## 2. Dataset

Dataset yang digunakan adalah **Rock Paper Scissors Images**.

Struktur dataset:

```text
rockpaperscissors/
│
├── paper/
├── rock/
└── scissors/
```

Dataset terdiri dari tiga kelas:

- paper → gambar tangan kertas
- rock → gambar tangan batu
- scissors → gambar tangan gunting

Jumlah data yang digunakan berdasarkan output program:

| Dataset | Jumlah |
|----------|---------|
| Training | 1751 gambar |
| Validation | 437 gambar |
| Total | 2188 gambar |

---

## 3. Persiapan Data Menggunakan ImageDataGenerator

Tahap preprocessing dilakukan menggunakan:

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```

Fungsi preprocessing yang digunakan:

### Rescaling

Nilai piksel gambar diubah dari:

```text
0–255
```

menjadi:

```text
0–1
```

menggunakan:

```python
rescale=1./255
```

### Pembagian Dataset

Dataset dibagi menjadi:

- Data training = 80%
- Data validation = 20%

### Konfigurasi Data

```python
target_size=(150,150)
batch_size=32
class_mode='categorical'
```

Keterangan:

- target_size → mengubah ukuran gambar menjadi 150×150 piksel
- batch_size → memproses 32 gambar setiap iterasi
- class_mode → klasifikasi multikelas

---

## 4. Arsitektur Model CNN

Model CNN dibangun menggunakan `Sequential()`.

Arsitektur model:

| Layer | Konfigurasi |
|---------|-------------|
| Conv2D | 32 filter (3×3), ReLU |
| MaxPooling2D | (2×2) |
| Conv2D | 64 filter (3×3), ReLU |
| MaxPooling2D | (2×2) |
| Conv2D | 128 filter (3×3), ReLU |
| MaxPooling2D | (2×2) |
| Flatten | - |
| Dense | 512 neuron, ReLU |
| Dense | 3 neuron, Softmax |

Implementasi model:

```python
model = Sequential([

Conv2D(
32,(3,3),
activation='relu',
input_shape=(150,150,3)
),

MaxPooling2D(2,2),

Conv2D(
64,(3,3),
activation='relu'
),

MaxPooling2D(2,2),

Conv2D(
128,(3,3),
activation='relu'
),

MaxPooling2D(2,2),

Flatten(),

Dense(
512,
activation='relu'
),

Dense(
3,
activation='softmax'
)

])
```

Penjelasan layer:

**Conv2D**

Mengekstraksi fitur dari gambar menggunakan filter.

**MaxPooling2D**

Mengurangi dimensi feature map agar proses training lebih efisien.

**Flatten**

Mengubah data multidimensi menjadi vektor satu dimensi.

**Dense**

Melakukan proses klasifikasi berdasarkan fitur yang diperoleh.

---

## 5. Kompilasi Model

Model dikompilasi menggunakan:

```python
model.compile(

loss='categorical_crossentropy',
optimizer='adam',
metrics=['accuracy']

)
```

Konfigurasi:

**Loss Function**

```text
categorical_crossentropy
```

Digunakan untuk klasifikasi multi-kelas.

**Optimizer**

```text
adam
```

Digunakan untuk memperbarui bobot model.

**Metrics**

```text
accuracy
```

Digunakan untuk mengukur tingkat akurasi model.

---

## 6. Proses Training

Training dilakukan menggunakan:

```python
history=model.fit(

train_generator,
validation_data=validation_generator,
epochs=10

)
```

Parameter pelatihan:

| Parameter | Nilai |
|------------|--------|
| Epoch | 10 |
| Batch Size | 32 |

---

## 7. Hasil Training dan Evaluasi

Berdasarkan output program:

| Parameter | Hasil |
|------------|--------|
| Validation Accuracy | 97.03% |
| Validation Loss | 0.0978 |

Evaluasi dilakukan menggunakan:

```python
val_loss,val_acc=model.evaluate(
validation_generator
)

print(
f'Validation loss:{val_loss},
Validation accuracy:{val_acc}'
)
```

Model memperoleh akurasi validasi yang tinggi sehingga dapat mengklasifikasikan gambar dengan baik.

---

## 8. Prediksi Model

Prediksi dilakukan menggunakan:

```python
predictions=model.predict(
validation_generator
)

print(predictions)
```

Contoh output:

```python
[[2.5656090e-07 4.7995679e-17 9.9999970e-01]

[6.4051501e-06 7.7203853e-19 9.9999350e-01]

[3.5617288e-12 9.7182449e-24 9.9999994e-01]]
```

Nilai terbesar menunjukkan kelas hasil prediksi.

---

## 9. Cara Menjalankan Program

Install dependency:

```bash
pip install tensorflow numpy pandas
```

Ekstrak dataset:

```bash
rockpaperscissors.zip
```

Pastikan struktur folder:

```text
rockpaperscissors/
├── paper/
├── rock/
└── scissors/
```

Buka file notebook:

```text
main.ipynb
```

Kemudian jalankan seluruh cell menggunakan Google Colab atau Jupyter Notebook.

---

Nama : Edgina Syafa  
NIM : H1D024046  
Shift : F
Shift KRS : B
