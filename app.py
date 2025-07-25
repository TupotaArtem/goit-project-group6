import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

st.title("Прогноз відтоку клієнтів телеком-компанії")

with open("models/best_model_xgboost.pkl", "rb") as f:
    model = pickle.load(f)

features = [
    'is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
    'bill_avg', 'reamining_contract', 'service_failure_count',
    'download_avg', 'upload_avg', 'download_over_limit', 'contract_unknown'
]

tab1, tab2 = st.tabs(["Індивідуальний прогноз", "Масова обробка (CSV)"])

# ==== Вкладка 1: Індивідуальний прогноз ====

with tab1:
    st.header("Введіть дані нового клієнта:")

    input_values = [
        st.number_input("Чи підписаний на телебачення? (1 = так, 0 = ні)", 0, 1),
        st.number_input("Чи підписаний на фільми? (1 = так, 0 = ні)", 0, 1),
        st.number_input("Тривалість користування послугою (місяці)", 0.0, 200.0),
        st.number_input("Середній чек", 0.0, 1000.0),
        st.number_input("Залишок контракту (місяці)", 0.0, 200.0),
        st.number_input("К-сть збоїв сервісу", 0.0, 100.0),
        st.number_input("Середня швидкість скачування (Mbps)", 0.0, 1000.0),
        st.number_input("Середня швидкість відвантаження (Mbps)", 0.0, 1000.0),
        st.number_input("Перевищення ліміту (GB)", 0.0, 1000.0),
        st.number_input("Невідомий тип контракту (1 = так, 0 = ні)", 0, 1)
    ]
    input_data_df = pd.DataFrame([dict(zip(features, input_values))])

    if st.button("Прогнозувати", key='single'):
        prob = model.predict_proba(input_data_df)[0, 1]
        st.write(f"**Ймовірність відтоку:** {prob:.2%}")
        if prob > 0.5:
            st.error("Клієнт має ВИСОКУ ймовірність відтоку.")
        else:
            st.success("Клієнт має НИЗЬКУ ймовірність відтоку.")

        st.progress(float(prob))

        # --- Візуалізація ---
        st.subheader("Візуалізація ймовірності відтоку")
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.barh(['Клієнт'], [prob], color='tomato' if prob > 0.5 else 'skyblue')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Ймовірність відтоку')
        ax.set_title('Горизонтальна шкала ймовірності')
        for i, v in enumerate([prob]):
            ax.text(v + 0.02, i, f"{v:.2%}", color='black', va='center', fontweight='bold')
        st.pyplot(fig)

# ==== Вкладка 2: Масова обробка (CSV) ====
with tab2:
    st.header("Масова обробка клієнтів (CSV)")

    uploaded_file = st.file_uploader("Завантажте CSV-файл з клієнтами", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Перші 5 записів із файлу:", df.head())

        df_preds = df[features].copy()
        preds = model.predict_proba(df_preds)[:, 1]
        df['churn_proba'] = preds
        df['churn_label'] = (preds > 0.5).astype(int)
        st.write("Результати прогнозу (перші 5):", df.head())
        st.dataframe(df)

        csv = df.to_csv(index=False).encode()
        st.download_button("Завантажити результати (CSV)", data=csv, file_name="results.csv")

        # --- Візуалізація ---
        st.subheader("Візуалізація результатів")
        labels = ['Не піде', 'Піде']
        sizes = [(df['churn_label'] == 0).sum(), (df['churn_label'] == 1).sum()]
        fig1, ax1 = plt.subplots()
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'tomato'])
        ax1.axis('equal')
        st.pyplot(fig1)

        st.subheader("Розподіл ймовірностей відтоку")
        fig2, ax2 = plt.subplots()
        ax2.hist(df['churn_proba'], bins=20, color='lightgreen', edgecolor='black')
        ax2.set_xlabel('Ймовірність відтоку')
        ax2.set_ylabel('Кількість клієнтів')
        st.pyplot(fig2)