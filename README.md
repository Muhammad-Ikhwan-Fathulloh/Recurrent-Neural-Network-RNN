# **Materi Lengkap RNN, LSTM, dan GRU dengan PyTorch**

## **1. Pendahuluan: Mengapa Kita Butuh Model Recurrent (Berulang)?**

Banyak jenis data di dunia nyata berbentuk **urutan (sequential data)**:

* Teks (urutan kata)
* Suara (urutan getaran suara)
* Sinyal sensor (urutan waktu)
* Harga saham (urutan nilai per hari)

Masalah utama model biasa (seperti feed-forward neural network) adalah:

> Mereka menganggap setiap input **tidak saling berhubungan**, padahal dalam data urutan, **urutan waktu sangat penting**.

---

### **Analogi**

Bayangkan kamu membaca kalimat:

> “Dia sedang berjalan di …”

Untuk menebak kata selanjutnya (“jalan”), kamu perlu **ingat kata-kata sebelumnya**.
Jadi, model harus punya **memori jangka pendek** tentang input sebelumnya.

Di sinilah **Recurrent Neural Network (RNN)** masuk.

---

## **2. Konsep Dasar RNN**

### **Struktur dan Rumus**

RNN memproses input **berulang** langkah demi langkah.

Pada setiap waktu `t`:

* Input: `x_t`
* Hidden state sebelumnya: `h_(t-1)`
* Hidden state sekarang: `h_t`

Persamaan dasarnya:
[
h_t = \tanh(W_x x_t + W_h h_{t-1} + b)
]
[
y_t = W_y h_t + c
]

Artinya:

* `h_t` menyimpan informasi dari masa lalu,
* `y_t` adalah output di waktu itu.

---

### **Analogi RNN**

RNN seperti **otak manusia saat mendengarkan kalimat**.
Setiap kata yang didengar menambah sedikit informasi pada pemahaman sebelumnya.

Namun, **otak manusia cepat lupa** — RNN juga begitu.
Ketika urutan data terlalu panjang, **informasi awal menghilang (vanishing gradient problem)**.

---

### **Implementasi RNN di PyTorch**

```python
import torch
import torch.nn as nn

# RNN sederhana
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

# Input (batch=5, sequence=3, fitur=10)
x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20)

output, hn = rnn(x, h0)

print(output.shape)  # (5, 3, 20)
print(hn.shape)      # (1, 5, 20)
```

---

## **3. LSTM (Long Short-Term Memory)**

### **Masalah RNN**

RNN kesulitan mengingat konteks jangka panjang karena:

* Gradien makin kecil setiap langkah (vanishing gradient)
* Informasi awal perlahan hilang

Untuk itu dibuatlah **LSTM** oleh Hochreiter & Schmidhuber (1997).

---

### **Konsep Inti**

LSTM menambahkan **memori jangka panjang (`c_t`)** dan **tiga gerbang (gate)** yang mengontrol aliran informasi.

1. **Forget Gate (f_t):**
   Memutuskan informasi apa yang dilupakan.
   [
   f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)
   ]

2. **Input Gate (i_t) & Candidate (g_t):**
   Menentukan informasi baru yang akan disimpan.
   [
   i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)
   ]
   [
   g_t = \tanh(W_g [h_{t-1}, x_t] + b_g)
   ]

3. **Cell State (c_t):**
   Menyimpan memori gabungan lama dan baru.
   [
   c_t = f_t * c_{t-1} + i_t * g_t
   ]

4. **Output Gate (o_t):**
   Mengontrol apa yang dikirim keluar.
   [
   o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)
   ]
   [
   h_t = o_t * \tanh(c_t)
   ]

---

### **Analogi Sederhana**

Bayangkan kamu seorang **penulis jurnal harian**:

* **Forget gate:** kamu pilih hal-hal yang tidak penting untuk diingat.
* **Input gate:** kamu catat hal-hal penting di buku harian.
* **Cell state:** buku harianmu, tempat semua memori tersimpan.
* **Output gate:** kamu ceritakan sebagian kepada temanmu besok.

---

### **Kelebihan LSTM**

* Dapat mengingat informasi jangka panjang.
* Mengatasi masalah vanishing gradient.
* Efektif untuk teks panjang, deret waktu panjang, dan audio.

---

### **Implementasi LSTM di PyTorch**

```python
import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20)
c0 = torch.zeros(1, 5, 20)

output, (hn, cn) = lstm(x, (h0, c0))

print(output.shape)  # (5, 3, 20)
```

---

## **4. GRU (Gated Recurrent Unit)**

### **Motivasi**

LSTM kuat, tapi **kompleks dan lambat** karena memiliki banyak gate dan dua state (`h_t`, `c_t`).

GRU (oleh Cho et al., 2014) dibuat sebagai **penyederhanaan dari LSTM**:

* Hanya 2 gerbang (update & reset)
* Hanya 1 state (`h_t`)

---

### **Rumus GRU**

1. **Update gate (z_t):**
   [
   z_t = \sigma(W_z [h_{t-1}, x_t])
   ]
   Menentukan seberapa banyak memori lama yang dipertahankan.

2. **Reset gate (r_t):**
   [
   r_t = \sigma(W_r [h_{t-1}, x_t])
   ]
   Menentukan seberapa banyak informasi lama yang dihapus.

3. **Hidden candidate (h̃_t):**
   [
   \tilde{h_t} = \tanh(W [r_t * h_{t-1}, x_t])
   ]

4. **Hidden state baru:**
   [
   h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h_t}
   ]

---

### **Analogi GRU**

Bayangkan kamu sedang **belajar lagu baru**:

* **Reset gate:** memutuskan apakah melupakan hafalan lama.
* **Update gate:** seberapa banyak kamu mengganti hafalan lama dengan yang baru.

GRU seperti LSTM tapi **lebih cepat dan hemat daya hitung**.

---

### **Implementasi GRU di PyTorch**

```python
import torch
import torch.nn as nn

gru = nn.GRU(input_size=10, hidden_size=20, num_layers=1, batch_first=True)

x = torch.randn(5, 3, 10)
h0 = torch.zeros(1, 5, 20)

output, hn = gru(x, h0)

print(output.shape)  # (5, 3, 20)
```

---

## **5. Perbandingan Lengkap**

| Aspek           | **RNN**     | **LSTM**                       | **GRU**                  |
| --------------- | ----------- | ------------------------------ | ------------------------ |
| Jumlah Gate     | Tidak ada   | 3 gate (Forget, Input, Output) | 2 gate (Update, Reset)   |
| State           | 1 (h_t)     | 2 (h_t, c_t)                   | 1 (h_t)                  |
| Kompleksitas    | Rendah      | Tinggi                         | Sedang                   |
| Kecepatan       | Cepat       | Lambat                         | Lebih cepat dari LSTM    |
| Daya Ingat      | Pendek      | Panjang                        | Menengah – panjang       |
| Cocok untuk     | Data pendek | Teks panjang, time series      | Aplikasi real-time, teks |
| Konsumsi Memori | Rendah      | Tinggi                         | Lebih hemat dari LSTM    |

---

## **6. Contoh Kasus Nyata: Prediksi Deret Waktu (Time Series)**

Kita akan membuat **model sederhana** untuk memprediksi angka berikutnya dari deret waktu.

### **Langkah-langkah**

1. Buat data urutan sinusoidal.
2. Masukkan ke model (RNN, LSTM, atau GRU).
3. Latih model untuk memprediksi nilai berikutnya.

---

### **Implementasi PyTorch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Data dummy: sin wave
x = np.linspace(0, 100, 1000)
y = np.sin(x)

# Siapkan data dalam bentuk (sequence_length, 1)
def create_dataset(data, seq_length=20):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i+seq_length])
        ys.append(data[i+seq_length])
    return np.array(xs), np.array(ys)

seq_length = 20
X, Y = create_dataset(y, seq_length)

X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (samples, seq, feature)
Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1)

# Model LSTM sederhana
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training
for epoch in range(100):
    output = model(X)
    loss = criterion(output, Y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

---

## **7. Kesimpulan**

| Model    | Ciri Utama                            | Gunakan Ketika                             |
| -------- | ------------------------------------- | ------------------------------------------ |
| **RNN**  | Sederhana, cepat, mudah dilatih       | Urutan pendek dan ringan                   |
| **LSTM** | Punya memori panjang, paling stabil   | Urutan panjang (teks, time series panjang) |
| **GRU**  | Lebih cepat dari LSTM tapi tetap kuat | Real-time dan perangkat terbatas           |

---

## **8. Penutup**

Jika diibaratkan:

* **RNN** seperti **orang yang mudah lupa** — bagus untuk pembicaraan singkat.
* **LSTM** seperti **orang yang menulis catatan harian** — mampu mengingat banyak hal.
* **GRU** seperti **orang yang punya ingatan kuat tanpa harus menulis** — efisien dan cepat.
