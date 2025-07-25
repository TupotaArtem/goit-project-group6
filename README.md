# 📈 Churn Prediction App

Інтерактивний застосунок для прогнозування відтоку клієнтів телекомунікаційної компанії на базі моделей машинного навчання.  
Реалізовано інтерфейс для одиночного та масового прогнозу (через CSV), запуск через Docker Compose.

## 🚀 Швидкий старт

1. Клонування репозиторію
   
   `git clone https://github.com/TupotaArtem/goit-project-group6`  
   `cd goit-project-group6`

3. Запуск у Docker
   
   `docker-compose up --build`

4. Відкрити у браузері
   
   `http://localhost:8501`

### 🛠️ Структура репозиторію

    ├── app.py
    ├── docker-compose.yml
    ├── Dockerfile
    ├── models
    │   └── best_model_xgboost.pkl
    ├── requirements.txt
    └── README.md
    
### 💻 Особливості роботи з інтерфейсом

•	Індивідуальний прогноз: введіть дані клієнта, натисніть “Прогнозувати” — отримайте ймовірність відтоку та візуалізацію.  

•	Масова обробка (CSV): завантажте CSV-файл, отримаєте ймовірності для всіх клієнтів, графіки, кнопку для збереження результатів.

### 📄 Формат CSV для масового прогнозу

    is_tv_subscriber,is_movie_package_subscriber,subscription_age,bill_avg,reamining_contract,service_failure_count,download_avg,upload_avg,download_over_limit,contract_unknown
    1,0,12,25.5,8,0,50.0,10.0,1.5,0
    0,1,6,15.0,3,1,20.0,5.0,0.0,1
    ...

•	Важливо: усі назви стовпців та порядок — як у прикладі!

### ⚙️ Налаштування (опціонально)  
    
•	Файл моделі має лежати в `models/`.  

•	Якщо потрібно оновити модель, просто замініть файл і перезапустіть сервіс.