import streamlit as st
import pickle
import numpy as np

with open ('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Прогноз відтоку клієнта телеком-компанії")

st.write("Введіть дані нового клієнта:")

is_tv_subscriber = st.number_input("Чи підписаний на телебачення? (1 = так, 0 = ні)", 0, 1)
is_movie_package_subscriber = st.number_input("Чи підписаний на фільми? (1 = так, 0 = ні)", 0, 1)
subscription_age = st.number_input("Тривалість користування послугою (місяці)", 0.0, 200.0)
bill_avg = st.number_input("Середній чек", 0.0, 1000.0)
reamining_contract = st.number_input("Залишок контракту (місяці)", 0.0, 200.0)
service_failure_count = st.number_input("К-сть збоїв сервісу", 0.0, 100.0)
download_avg = st.number_input("Середня швидкість скачування (Mbps)", 0.0, 1000.0)
upload_avg = st.number_input("Середня швидкість відвантаження (Mbps)", 0.0, 1000.0)
download_over_limit = st.number_input("Перевищення ліміту (GB)", 0.0, 1000.0)
contract_unknown = st.number_input("Невідомий тип контракту (1 = так, 0 = ні)", 0, 1)

input_data = np.array([[
    is_tv_subscriber,
    is_movie_package_subscriber,
    subscription_age,
    bill_avg,
    reamining_contract,
    service_failure_count,
    download_avg,
    upload_avg,
    download_over_limit,
    contract_unknown
]])

if st.button("Прогнозувати"):
    prob = model.predict_proba(input_data)[0, 1]
    st.write(f"**Ймовірність відтоку:** {prob:.2%}")
    if prob > 0.5:
        st.error("Клієнт має ВИСОКУ ймовірність відтоку.")
    else:
        st.success("Клієнт має НИЗЬКУ ймовірність відтоку.")

    st.progress(float(prob))


st.title("Docker-тест: Всё працює!")
st.write("Вітаємо у додатку, запущеному в Docker-контейнері.")
