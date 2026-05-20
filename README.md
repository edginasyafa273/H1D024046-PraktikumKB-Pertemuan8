# H1D024046-PraktikumKB-Pertemuan8

Repositori pengumpulan tugas praktikum mata kuliah Kecerdasan Buatan (Pertemuan 8).

## Implementasi Convolutional Neural Network (CNN) untuk Klasifikasi Gambar Rock-Paper-Scissors

Proyek ini berisi implementasi model Deep Learning menggunakan metode **Convolutional Neural Network (CNN)** dengan TensorFlow/Keras untuk melakukan klasifikasi gambar tangan menjadi tiga kategori, yaitu:

- Rock (Batu)
- Paper (Kertas)
- Scissors (Gunting)

Dataset yang digunakan berupa kumpulan citra Rock-Paper-Scissors yang diproses menggunakan **ImageDataGenerator**, kemudian dilatih menggunakan arsitektur CNN dengan beberapa layer konvolusi dan pooling.

---

## Library yang Digunakan

Program menggunakan beberapa library Python berikut:

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile
```

Library yang digunakan memiliki fungsi sebagai berikut:

- **NumPy** → manipulasi array numerik
- **Pandas** → pengolahan data
- **TensorFlow/Keras** → membangun dan melatih model CNN
- **ImageDataGenerator** → preprocessing data gambar
- **zipfile** → ekstraksi dataset zip

---

## Dataset

Dataset yang digunakan adalah dataset **Rock-Paper-Scissors** yang berisi:

- Total gambar: **2188**
- Data training: **1751 gambar**
- Data validasi: **437 gambar**
- Jumlah kelas: **3**

Struktur dataset:

```

rockpaperscissors/
│
├── paper/
├── rock/
└── scissors/

```

---

## Tahapan Program

### 1. Ekstraksi Dataset

Dataset yang masih berbentuk file ZIP diekstrak terlebih dahulu menggunakan library `zipfile`.

```python
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

---

### 2. Preprocessing Data

Data diproses menggunakan `ImageDataGenerator`.

Parameter yang digunakan:

- Rescale = 1/255
- Validation split = 20%
- Target size = (150,150)
- Batch size = 32

```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)
```

---

### 3. Pembuatan Model CNN

Arsitektur CNN yang digunakan:

| Layer | Output |
|---------|----------|
| Conv2D (32 filter) | (148,148,32) |
| MaxPooling2D | (74,74,32) |
| Conv2D (64 filter) | (72,72,64) |
| MaxPooling2D | (36,36,64) |
| Conv2D (128 filter) | (34,34,128) |
| MaxPooling2D | (17,17,128) |
| Flatten | 36992 |
| Dense (512 neuron) | 512 |
| Dense Output (3 neuron) | 3 |

Kode model:

```python
model = Sequential([
    Conv2D(32,(3,3),activation='relu',
           input_shape=(150,150,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(128,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(512,activation='relu'),
    Dense(3,activation='softmax')
])
```

---

## Kompilasi Model

Model menggunakan:

- Loss function: `categorical_crossentropy`
- Optimizer: `adam`
- Metrics: `accuracy`

```python
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
```

---

## Training Model

Model dilatih sebanyak:

- Epoch = 10
- Batch size = 32

```python
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)
```

---

## Hasil Training

Hasil pelatihan menunjukkan peningkatan akurasi yang cukup baik.

| Epoch | Accuracy | Validation Accuracy |
|---------|------------|----------------------|
| 1 | 64.82% | 88.33% |
| 5 | 99.20% | 96.11% |
| 10 | 100% | 97.03% |

Hasil evaluasi akhir:

```python
Validation loss: 0.0978
Validation accuracy: 0.9703
```

---

## Hasil Prediksi

Model menghasilkan probabilitas prediksi untuk masing-masing kelas:

Contoh output:

```python
[[2.5656090e-07 4.7995679e-17 9.9999970e-01]
 [6.4051501e-06 7.7203853e-19 9.9999350e-01]
 [3.5617288e-12 9.7182449e-24 9.9999994e-01]]
```

Output menunjukkan probabilitas klasifikasi pada:

- Rock
- Paper
- Scissors

Nilai probabilitas tertinggi menunjukkan hasil prediksi kelas gambar.

---

## Kesimpulan

Berdasarkan hasil implementasi CNN untuk klasifikasi gambar Rock-Paper-Scissors, model berhasil mencapai:

- Training Accuracy = **100%**
- Validation Accuracy = **97.03%**

Hasil tersebut menunjukkan bahwa model CNN mampu mempelajari pola gambar dengan baik dan melakukan klasifikasi dengan tingkat akurasi yang tinggi.

---

**Nama:** Edgina Syafa  
**NIM:** H1D024046  
