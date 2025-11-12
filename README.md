# **Materi Lengkap: RNN, LSTM, dan GRU dengan PyTorch**

---

## **1. Pengantar**

Banyak data di dunia nyata berbentuk **urutan (sequence)** seperti:

* Kalimat dalam teks
* Suara manusia
* Data sensor IoT
* Pergerakan harga saham

Model konvensional seperti **Feed Forward Neural Network (FFNN)** tidak mampu menangkap hubungan antar waktu, karena **setiap input dianggap terpisah**.
Untuk menangani urutan, kita butuh model yang bisa **mengingat konteks sebelumnya** — inilah alasan lahirnya **Recurrent Neural Network (RNN)** dan turunannya **LSTM** serta **GRU**.

---

## **2. RNN (Recurrent Neural Network)**

### **Konsep Dasar**

RNN memproses data **langkah demi langkah** (misalnya kata demi kata).
Pada setiap langkah, model menyimpan informasi sebelumnya ke dalam **hidden state**, lalu menggunakannya untuk memproses input berikutnya.

Bayangkan seperti **orang yang mendengarkan cerita** — setiap kalimat yang didengar akan memengaruhi pemahaman terhadap kalimat berikutnya.

---

### **Kelebihan RNN**

* Struktur sederhana dan mudah dipahami.
* Cepat dilatih untuk urutan pendek.
* Cocok untuk tugas sederhana seperti prediksi huruf atau kata pendek.

### **Kelemahan RNN**

* **Sulit mengingat konteks jangka panjang** (mudah lupa).
* **Vanishing gradient problem** membuat pembelajaran menjadi tidak stabil.
* Kurang efisien untuk data panjang seperti paragraf atau sinyal panjang.

---

### **Contoh Kode RNN di PyTorch**

```python
import torch
import torch.nn as nn

# RNN sederhana
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Input: (batch, sequence_length, fitur)
x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20)

output, hn = rnn(x, h0)

print("Output shape:", output.shape)
print("Hidden state shape:", hn.shape)
```

---

## **3. LSTM (Long Short-Term Memory)**

### **Konsep Dasar**

LSTM merupakan **penyempurnaan dari RNN**.
Model ini dapat **mengingat informasi penting lebih lama** dan **melupakan informasi yang tidak relevan**.
Ia memiliki sistem “memori internal” yang membantu mengatasi masalah lupa pada RNN.

Bayangkan seperti **orang yang menulis catatan harian**:

* Informasi penting dicatat (agar tidak lupa).
* Hal tidak penting diabaikan.
* Bisa melihat kembali catatan lama jika dibutuhkan.

---

### **Kelebihan LSTM**

* Dapat mengingat konteks jangka panjang.
* Stabil saat menangani urutan panjang.
* Efektif untuk teks, audio, atau time series.

### **Kelemahan LSTM**

* Struktur lebih kompleks dari RNN.
* Proses pelatihan lebih lama dan butuh memori besar.
* Tidak selalu efisien untuk aplikasi real-time.

---

### **Contoh Kode LSTM di PyTorch**

```python
import torch
import torch.nn as nn

# LSTM sederhana
lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20)
c0 = torch.zeros(1, 5, 20)

output, (hn, cn) = lstm(x, (h0, c0))

print("Output shape:", output.shape)
print("Hidden state shape:", hn.shape)
print("Cell state shape:", cn.shape)
```

---

## **4. GRU (Gated Recurrent Unit)**

### **Konsep Dasar**

GRU adalah **versi lebih sederhana dari LSTM**.
Ia bekerja hampir sama, tetapi **menggabungkan sistem memori dan hidden state menjadi satu**, sehingga **lebih ringan dan cepat**.

Bayangkan GRU seperti **orang yang punya daya ingat kuat tanpa perlu menulis catatan** — tetap bisa mengingat hal penting tanpa mekanisme yang rumit.

---

### **Kelebihan GRU**

* Lebih cepat dari LSTM.
* Performa hampir sama dengan LSTM.
* Lebih hemat memori dan mudah dilatih.

### **Kelemahan GRU**

* Kurang akurat pada urutan yang sangat panjang.
* Tidak sefleksibel LSTM untuk mengatur memori.

---

### **Contoh Kode GRU di PyTorch**

```python
import torch
import torch.nn as nn

# GRU sederhana
gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20)

output, hn = gru(x, h0)

print("Output shape:", output.shape)
print("Hidden state shape:", hn.shape)
```

---

## **5. Perbandingan RNN vs LSTM**

| Aspek             | **RNN**                           | **LSTM**                                |
| ----------------- | --------------------------------- | --------------------------------------- |
| Struktur          | Sederhana                         | Kompleks (punya memori tambahan)        |
| Kecepatan Latih   | Cepat                             | Lebih lambat                            |
| Daya Ingat        | Pendek                            | Panjang                                 |
| Stabilitas        | Kurang stabil pada urutan panjang | Stabil pada urutan panjang              |
| Penggunaan Memori | Rendah                            | Lebih tinggi                            |
| Masalah Umum      | Vanishing gradient                | Lebih tahan terhadap vanishing gradient |
| Cocok Untuk       | Data pendek, sederhana            | Teks panjang, data sekuensial kompleks  |

---

### **Penjelasan Singkat**

* **RNN** mudah dilatih tapi cepat “lupa” konteks lama.
* **LSTM** memiliki mekanisme penyimpanan memori, sehingga dapat mengingat konteks panjang, tetapi lebih lambat karena struktur lebih rumit.

---

## **6. Perbandingan Lengkap (RNN, LSTM, GRU)**

| Aspek       | **RNN**       | **LSTM**      | **GRU**                |
| ----------- | ------------- | ------------- | ---------------------- |
| Struktur    | Sederhana     | Kompleks      | Sederhana tapi efisien |
| Kecepatan   | Cepat         | Paling lambat | Lebih cepat dari LSTM  |
| Daya Ingat  | Pendek        | Panjang       | Menengah–panjang       |
| Stabilitas  | Kurang stabil | Stabil        | Stabil                 |
| Komputasi   | Rendah        | Tinggi        | Sedang                 |
| Cocok Untuk | Urutan pendek | Data panjang  | Aplikasi real-time     |
| Kekurangan  | Cepat lupa    | Lambat, berat | Kadang kurang presisi  |

---

## **7. Contoh Aplikasi Mini: Prediksi Deret Waktu dengan LSTM**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data sin wave
x = np.linspace(0, 100, 1000)
y = np.sin(x)

def create_dataset(data, seq_len=20):
    X, Y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len])
    return np.array(X), np.array(Y)

seq_len = 20
X, Y = create_dataset(y, seq_len)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

# Model LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(50):
    output = model(X)
    loss = criterion(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## **8. Kesimpulan Akhir**

| Model    | Kelebihan                                | Kelemahan                                    | Cocok Untuk                       |
| -------- | ---------------------------------------- | -------------------------------------------- | --------------------------------- |
| **RNN**  | Sederhana, cepat, mudah dipahami         | Mudah lupa konteks panjang                   | Urutan pendek dan ringan          |
| **LSTM** | Dapat mengingat konteks panjang          | Lambat dan kompleks                          | Teks panjang, time series         |
| **GRU**  | Cepat, hemat memori, performa mirip LSTM | Kurang fleksibel untuk urutan sangat panjang | Aplikasi real-time, teks menengah |

---

### **Analoginya**

* **RNN** → seperti orang yang mendengarkan cerita tapi cepat lupa bagian awal.
* **LSTM** → seperti orang yang menulis catatan penting agar tidak lupa.
* **GRU** → seperti orang dengan ingatan kuat tanpa harus mencatat.
