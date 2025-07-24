# Базовий образ з Python
FROM python:3.10-slim

# Встановлення робочої директорії в контейнері
WORKDIR /app

# Копіюємо файли з локального комп'ютера в контейнер
COPY . /app

# Встановлюємо залежності
RUN pip install --no-cache-dir -r requirements.txt

# Вказуємо команду для запуску застосунку
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
