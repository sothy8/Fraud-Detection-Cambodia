# Fraud-Detection-Cambodia

# Real‑Time AI‑Powered Fraud Detection System for Mobile Wallet Transactions in Cambodia 🇰🇭

> **Protect Cambodian mobile‑wallet users from scams and unauthorized transactions with an adaptive, real‑time machine‑learning engine.**

---

## ✨ Key Features
- **Real‑time detection** of anomalous P2P transfers, merchant payments, and top‑ups  
- **Adaptive ML models** (Isolation Forest / XGBoost) that learn each user’s normal behavior  
- **Streamlit dashboard** showing live transactions, fraud scores, and rule explanations  
- **Alerting module** (SMS / email simulation) for high‑risk events  
- **Extensible pipeline** ready for integration with Wing, Pi Pay, ABA Pay, etc.

---

## 🗂 Project Structure

```bash
fraud-detection-cambodia/
├── data/ # Simulated or real transaction CSVs
├── notebooks/ # EDA, feature engineering, model training
├── app/ # Streamlit dashboard & API endpoints
│ ├── main.py
│ └── utils.py
├── models/ # Saved model binaries (.pkl/.onnx)
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack
| Layer           | Tools / Libraries                                  |
|-----------------|----------------------------------------------------|
| **Data Sim**    | `python`, `faker`, `pandas`                         |
| **ML Engine**   | `scikit‑learn`, `xgboost`, `numpy`                  |
| **Dashboard**   | `streamlit`, `plotly`, `pandas`                     |
| **Alerts**      | `smtplib` / any SMS gateway API (simulated)         |
| **Deploy (Opt)**| `heroku`, `docker`, `gunicorn`                      |

---

## 🚀 Quick Start
1. **Clone** the repo  
   ```bash
   git clone https://github.com/<your‑user>/fraud-detection-cambodia.git
   cd fraud-detection-cambodia
   ```

2. **Install dependencies**
   ```bash
   python -m venv venv && source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Generate / load data**
   ```bash
   python scripts/simulate_transactions.py --out data/transactions.csv
   ```

4. **Train a model**
   ```bash
   python scripts/train_model.py
   ```

5. **Run the dashboard**
   ```bash
   # Method 1: Quick launcher (recommended)
   python run_dashboard.py
   
   # Method 2: Direct execution
   python app/main.py
   
   # Method 3: Auto-setup and start
   ./start.sh
   
   # Method 4: Management script
   ./manage.sh dashboard
   
   # Method 5: Traditional Streamlit
   streamlit run app/main.py
   ```

📈 Dataset Schema
| Column            | Type     | Description                              |
| ----------------- | -------- | ---------------------------------------- |
| `txn_id`          | string   | Unique transaction ID                    |
| `user_id`         | string   | Sender’s wallet identifier               |
| `recipient_id`    | string   | Receiver / merchant ID                   |
| `amount`          | float    | Transaction value in KHR                 |
| `txn_type`        | string   | `p2p` \| `merchant` \| `topup`           |
| `timestamp`       | datetime | Epoch ms                                 |
| `location_latlon` | string   | GPS pair (optional)                      |
| `device_id`       | string   | Hashed device fingerprint                |
| `label`\*         | int      | `1` = Fraud, `0` = Legit (if supervised) |
*Omit label for unsupervised mode.

🏗 System Architecture
```bash
                 +---------------------------+
                 |  Mobile‑Wallet Platform   |
                 +---------------------------+
                              |
                              v
               [Streaming/Batch Transaction Log]
                              |
                              v
+-------------+    Feature    +---------------+
|  Pre‑Proc   |  ───►  Store  |  ML Scoring   |
+-------------+               +---------------+
                              |
                              v
                    +----------------+
                    |  Alert Engine  |
                    +----------------+
                              |
                              v
                  +----------------------+
                  |  Streamlit Dashboard |
                  +----------------------+
```

🗓 Roadmap
- Simulated dataset generator
- Baseline Isolation Forest model
- XGBoost supervised pipeline (needs labeled data)
- Model drift monitoring
- Khmer‑localized UI / mobile‑friendly dashboard
- Dockerized deployment

🤝 Contributing
1. Fork the repo & create your feature branch (git checkout -b feat/awesome‑thing)
2. Commit your changes (git commit -am 'Add awesome thing')
3. Push to the branch (git push origin feat/awesome‑thing)
4. Open a Pull Request

📜 License
Distributed under the MIT License. See LICENSE for more information.

📬 Contact
Sothy Vandeth – <sothyvandeth8034@example.com>
Project Link: <https://github.com/sothy8/fraud-detection-cambodia>
Building a safer digital finance ecosystem for every Cambodian.
