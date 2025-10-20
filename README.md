# 🤖 Pro Trader AI - Bahasa Indonesia

AI Trader Profesional dengan kemampuan:
- Analisis Crypto (Binance API) & Forex (CSV)
- Scalping & Swing otomatis
- Auto-learning (RandomForest retrain tiap 50 sinyal)
- Analisis gambar chart (OCR)
- Auto logging ke trade_log.csv

## 🚀 Cara Deploy di Railway
1. Upload semua file ini ke repo GitHub.
2. Buka [https://railway.app](https://railway.app)
3. Klik **New Project → Deploy from GitHub repo**
4. Tunggu “Build success ✅”
5. Tes endpoint:

## 📡 Endpoint API Utama
| Endpoint | Fungsi |
|-----------|--------|
| `/health` | Cek apakah AI aktif |
| `/pro_signal` | Sinyal trend & entry utama |
| `/scalp_signal` | Sinyal scalping cepat |
| `/analyze_chart` | Analisis chart dari gambar |
| `/analyze_csv` | Analisis data forex (CSV) |
| `/learning_status` | Cek status AI learning |
| `/retrain_learning` | Paksa retrain model |
| `/logs` | Lihat semua sinyal tersimpan |

---

Dibuat oleh **Pro Trader AI (ID)** 🧠🇮🇩
