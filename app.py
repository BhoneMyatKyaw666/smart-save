"""STARTER FILE: fill in TODO 1-3, then rename to app.py (or copy into app.py).

ICT336 workshop: a savings website with 2 ML models (Kaggle data).

Tab 1  Fixed deposit -> supervised model (classification) predicts if a customer will open a fixed deposit
Tab 2  Saver group   -> unsupervised model (K-Means)       finds which group of customers a person is like
Tab 3  About         -> how the models were built

Data: Kaggle "Bank Marketing" and "Mall Customer Segmentation Data".
Run:  streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

# ---------- STEP 1: page title ----------
st.set_page_config(page_title="Smart Savings", page_icon="💰", layout="wide")
st.title("💰 Smart Savings")
st.caption("ICT336 workshop. Data from Kaggle, for teaching only, not financial advice.")


# ---------- STEP 2: load the saved models (only once) and make tabs ----------
@st.cache_resource
def load_models():
    # TODO 1: load "savings_model.joblib" and "saver_model.joblib" with joblib.load
    savings = joblib.load("savings_model.joblib")
    saver = joblib.load("saver_model.joblib")

    return savings, saver


@st.cache_data
def load_customers():
    return pd.read_csv("customers_with_groups.csv")


savings, saver = load_models()
customers = load_customers()
if savings is None or saver is None:
    st.info("Models not loaded yet: finish TODO 1.")
    st.stop()

tab_deposit, tab_group, tab_about = st.tabs(["🏦 Fixed deposit", "👥 Saver group", "ℹ️ About the models"])

# ---------- STEP 3: supervised model -> will this customer open a fixed deposit? ----------
with tab_deposit:
    st.subheader("Will this customer open a fixed deposit if the bank calls?")
    choices = savings["choices"]
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 17, 98, 40)
        job = st.selectbox("Job", choices["job"], index=choices["job"].index("admin."))
        marital = st.selectbox("Marital status", choices["marital"], index=choices["marital"].index("married"))
        education = st.selectbox("Education", choices["education"],
                                 index=choices["education"].index("university.degree"))
        default = st.selectbox("Has unpaid credit (default)?", choices["default"])
        housing = st.selectbox("Has a home loan?", choices["housing"], index=choices["housing"].index("yes"))
    with col2:
        loan = st.selectbox("Has a personal loan?", choices["loan"])
        campaign = st.slider("Calls in this campaign", 1, 20, 2)
        previous = st.slider("Calls in earlier campaigns", 0, 7, 0)
        poutcome = st.selectbox("Result of the last campaign", choices["poutcome"],
                                index=choices["poutcome"].index("nonexistent"))
        euribor3m = st.slider("Market interest rate (Euribor 3 months, %)", 0.6, 5.1, 4.9, step=0.1)

    customer = pd.DataFrame([{
        "age": age, "campaign": campaign, "previous": previous, "euribor3m": euribor3m,
        "job": job, "marital": marital, "education": education, "default": default,
        "housing": housing, "loan": loan, "poutcome": poutcome}])

    if st.button("Predict", type="primary"):
        # TODO 2: use savings["model"].predict_proba(...) on customer[savings["features"]]
        #         and take [0, 1] = the chance of "yes" for our one customer
        chance = savings["model"].predict_proba(
        customer[savings["features"]])[0, 1]
        st.metric("Chance of opening a fixed deposit", f"{chance * 100:.0f}%")
        st.progress(float(chance))
        if chance >= 0.7:
            st.success("High: call this customer first.")
        elif chance >= 0.5:
            st.warning("Medium: worth a call.")
        else:
            st.info("Low: send an SMS or leaflet instead of calling.")

# ---------- STEP 4: unsupervised model -> saver group ----------
with tab_group:
    st.subheader("Which group of customers is this person like?")
    col1, col2 = st.columns(2)
    with col1:
        income = st.slider("Annual income (thousand $)", 15, 140, 60)
    with col2:
        spending = st.slider("Spending score (1 = spends little, 100 = spends a lot)", 1, 100, 50)

    person = pd.DataFrame([{"Annual Income (k$)": income, "Spending Score (1-100)": spending}])

    if st.button("Find group", type="primary"):
        # TODO 3: use saver["model"].predict(...) on person[saver["features"]] and take [0]
        group = int(saver["model"].predict(
        person[saver["features"]]))
        st.success(f"This person is in group {group}: **{saver['names'][group]}**")
        st.info(f"Savings tip: {saver['tips'][group]}")

        st.write("Average customer in each group (this person is the last row):")
        table = saver["profile"].copy()
        table.index = [saver["names"][g] for g in table.index]
        table.loc["➡️ This person"] = person.iloc[0]
        st.dataframe(table.round(1))

        st.write("All 200 customers, coloured by group:")
        st.scatter_chart(customers, x="Annual Income (k$)", y="Spending Score (1-100)", color="group_name")

# ---------- STEP 5: explain the models ----------
with tab_about:
    st.subheader("1. Fixed deposit model (supervised)")
    st.write(f"Trained on 32,950 past calls with the answer (opened a fixed deposit or not). "
             f"Only 11% said yes, so we chose the model by F1, not accuracy. "
             f"Best model: **{savings['model_name']}**. Scores on 8,238 test calls:")
    st.dataframe(pd.DataFrame(savings["scores"]), hide_index=True)
    st.caption("Data: Kaggle, Bank Marketing (henriqueyamahata/bank-marketing).")

    st.subheader("2. Saver-group model (unsupervised)")
    st.write("K-Means found 5 groups in 200 customers using income and spending, with no answer column. "
             "We looked at the averages and gave each group a name and a savings tip:")
    st.dataframe(customers["group_name"].value_counts().rename("customers"))
    st.caption("Data: Kaggle, Mall Customer Segmentation Data (vjchoudhary7/customer-segmentation-tutorial-in-python).")
