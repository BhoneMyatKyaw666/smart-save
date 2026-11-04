
import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ==========================================
# PAGE SETTINGS
# ==========================================

st.set_page_config(
    page_title="Smart Savings",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Smart Savings")
st.caption(
    "ICT336 workshop. Machine learning demonstration, "
    "not financial advice."
)


# ==========================================
# DATA AND MODEL FILES
# ==========================================

SAVINGS_FILE = "savings_model.joblib"
SAVER_FILE = "saver_model.joblib"
CUSTOMERS_FILE = "customers_with_groups.csv"


# ==========================================
# TRAIN FIXED DEPOSIT MODEL
# ==========================================

def train_savings_model():

    np.random.seed(42)

    n = 5000

    jobs = [
        "admin.", "technician", "services", "management",
        "retired", "blue-collar", "unemployed",
        "entrepreneur", "student", "housemaid"
    ]

    marital_options = ["married", "single", "divorced"]

    education_options = [
        "basic.4y", "basic.6y", "basic.9y",
        "high.school", "university.degree"
    ]

    yes_no = ["yes", "no", "unknown"]

    poutcomes = ["nonexistent", "failure", "success"]

    df = pd.DataFrame({
        "age": np.random.randint(18, 90, n),
        "campaign": np.random.randint(1, 15, n),
        "previous": np.random.randint(0, 6, n),
        "euribor3m": np.random.uniform(0.6, 5.1, n),

        "job": np.random.choice(jobs, n),
        "marital": np.random.choice(marital_options, n),
        "education": np.random.choice(education_options, n),
        "default": np.random.choice(yes_no, n),
        "housing": np.random.choice(yes_no, n),
        "loan": np.random.choice(yes_no, n),
        "poutcome": np.random.choice(poutcomes, n)
    })

    # Demonstration target generated from input features.
    score = (
        (df["age"] > 55).astype(int) * 0.5
        + (df["poutcome"] == "success").astype(int) * 2.0
        + (df["previous"] > 0).astype(int) * 0.5
        + (df["euribor3m"] < 2.0).astype(int) * 0.8
        + (df["job"] == "retired").astype(int) * 0.4
        - (df["campaign"] > 5).astype(int) * 0.5
        + np.random.normal(0, 0.8, n)
    )

    target = (score > 1.5).astype(int)

    features = list(df.columns)

    categorical = [
        "job", "marital", "education", "default",
        "housing", "loan", "poutcome"
    ]

    numerical = [
        "age", "campaign", "previous", "euribor3m"
    ]

    preprocessor = ColumnTransformer([
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical
        ),
        ("numerical", StandardScaler(), numerical)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight="balanced"
            )
        )
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    scores = [{
        "Accuracy": round(accuracy_score(y_test, predictions), 3),
        "F1 Score": round(f1_score(y_test, predictions), 3)
    }]

    choices = {
        "job": jobs,
        "marital": marital_options,
        "education": education_options,
        "default": yes_no,
        "housing": yes_no,
        "loan": yes_no,
        "poutcome": poutcomes
    }

    result = {
        "model": model,
        "features": features,
        "choices": choices,
        "model_name": "Random Forest",
        "scores": scores
    }

    joblib.dump(result, SAVINGS_FILE)

    return result


# ==========================================
# TRAIN SAVER GROUP MODEL
# ==========================================

def train_saver_model():

    np.random.seed(42)

    n = 200

    income = np.random.randint(15, 141, n)
    spending = np.random.randint(1, 101, n)

    customers = pd.DataFrame({
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending
    })

    features = [
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    scaler = StandardScaler()

    scaled = scaler.fit_transform(customers[features])

    model = KMeans(
        n_clusters=5,
        random_state=42,
        n_init=10
    )

    groups = model.fit_predict(scaled)

    customers["group"] = groups

    names = {}
    tips = {}

    for group in range(5):

        group_data = customers[customers["group"] == group]

        avg_income = group_data["Annual Income (k$)"].mean()
        avg_spending = group_data["Spending Score (1-100)"].mean()

        if avg_income >= 70 and avg_spending >= 60:
            names[group] = "High Income Spenders"
            tips[group] = "Consider setting a monthly savings target."

        elif avg_income >= 70 and avg_spending < 60:
            names[group] = "High Income Savers"
            tips[group] = "Consider investing part of your savings."

        elif avg_income < 40:
            names[group] = "Budget Conscious"
            tips[group] = "Start with a small emergency fund."

        elif avg_spending >= 60:
            names[group] = "Frequent Spenders"
            tips[group] = "Track your spending and set a budget."

        else:
            names[group] = "Balanced Customers"
            tips[group] = "Continue maintaining a balanced budget."

    customers["group_name"] = customers["group"].map(names)

    profile = customers.groupby("group")[features].mean()

    result = {
        "model": model,
        "scaler": scaler,
        "features": features,
        "names": names,
        "tips": tips,
        "profile": profile
    }

    customers.to_csv(CUSTOMERS_FILE, index=False)

    joblib.dump(result, SAVER_FILE)

    return result, customers


# ==========================================
# LOAD MODELS SAFELY
# ==========================================

@st.cache_resource
def load_models():

    try:
        savings = joblib.load(SAVINGS_FILE)
        saver = joblib.load(SAVER_FILE)

        # Validate the expected model structure.
        required_savings = [
            "model", "features", "choices",
            "model_name", "scores"
        ]

        required_saver = [
            "model", "scaler", "features",
            "names", "tips", "profile"
        ]

        if not all(key in savings for key in required_savings):
            raise ValueError("Invalid savings model file")

        if not all(key in saver for key in required_saver):
            raise ValueError("Invalid saver model file")

        return savings, saver

    except Exception:

        st.warning(
            "Old or incompatible model files detected. "
            "Training fresh demonstration models..."
        )

        savings = train_savings_model()

        saver, _ = train_saver_model()

        return savings, saver


# ==========================================
# LOAD CUSTOMER DATA
# ==========================================

@st.cache_data
def load_customers():

    if os.path.exists(CUSTOMERS_FILE):

        try:
            return pd.read_csv(CUSTOMERS_FILE)
        except Exception:
            pass

    _, customers = train_saver_model()

    return customers


# ==========================================
# INITIALIZE
# ==========================================

try:

    savings, saver = load_models()
    customers = load_customers()

except Exception as e:

    st.error(f"Application initialization failed: {e}")
    st.stop()


# ==========================================
# TABS
# ==========================================

tab_deposit, tab_group, tab_about = st.tabs([
    "🏦 Fixed deposit",
    "👥 Saver group",
    "ℹ️ About the models"
])


# ==========================================
# TAB 1: FIXED DEPOSIT PREDICTION
# ==========================================

with tab_deposit:

    st.subheader(
        "Will this customer open a fixed deposit if the bank calls?"
    )

    choices = savings["choices"]

    col1, col2 = st.columns(2)

    with col1:

        age = st.slider("Age", 17, 98, 40)

        job = st.selectbox(
            "Job",
            choices["job"],
            index=choices["job"].index("admin.")
        )

        marital = st.selectbox(
            "Marital status",
            choices["marital"],
            index=choices["marital"].index("married")
        )

        education = st.selectbox(
            "Education",
            choices["education"],
            index=choices["education"].index("university.degree")
        )

        default = st.selectbox(
            "Has unpaid credit (default)?",
            choices["default"]
        )

        housing = st.selectbox(
            "Has a home loan?",
            choices["housing"],
            index=choices["housing"].index("yes")
        )

    with col2:

        loan = st.selectbox(
            "Has a personal loan?",
            choices["loan"]
        )

        campaign = st.slider(
            "Calls in this campaign", 1, 20, 2
        )

        previous = st.slider(
            "Calls in earlier campaigns", 0, 7, 0
        )

        poutcome = st.selectbox(
            "Result of the last campaign",
            choices["poutcome"],
            index=choices["poutcome"].index("nonexistent")
        )

        euribor3m = st.slider(
            "Market interest rate (Euribor 3 months, %)",
            0.6, 5.1, 4.9, step=0.1
        )

    customer = pd.DataFrame([{
        "age": age,
        "campaign": campaign,
        "previous": previous,
        "euribor3m": euribor3m,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "poutcome": poutcome
    }])

    if st.button("Predict", type="primary"):

        chance = savings["model"].predict_proba(
            customer[savings["features"]]
        )[0, 1]

        st.metric(
            "Chance of opening a fixed deposit",
            f"{chance * 100:.1f}%"
        )

        st.progress(float(chance))

        if chance >= 0.7:
            st.success("High: call this customer first.")

        elif chance >= 0.5:
            st.warning("Medium: worth a call.")

        else:
            st.info(
                "Low: send an SMS or leaflet instead of calling."
            )


# ==========================================
# TAB 2: SAVER GROUP
# ==========================================

with tab_group:

    st.subheader(
        "Which group of customers is this person like?"
    )

    col1, col2 = st.columns(2)

    with col1:

        income = st.slider(
            "Annual income (thousand $)",
            15, 140, 60
        )

    with col2:

        spending = st.slider(
            "Spending score (1 = spends little, 100 = spends a lot)",
            1, 100, 50
        )

    person = pd.DataFrame([{
        "Annual Income (k$)": income,
        "Spending Score (1-100)": spending
    }])

    if st.button("Find group", type="primary"):

        scaled_person = saver["scaler"].transform(
            person[saver["features"]]
        )

        group = int(
            saver["model"].predict(scaled_person)[0]
        )

        st.success(
            f"This person is in group {group}: "
            f"**{saver['names'][group]}**"
        )

        st.info(
            f"Savings tip: {saver['tips'][group]}"
        )

        st.write(
            "Average customer in each group "
            "(this person is the last row):"
        )

        table = saver["profile"].copy()

        table.index = [
            saver["names"][g] for g in table.index
        ]

        table.loc["➡️ This person"] = person.iloc[0]

        st.dataframe(table.round(1))

        st.write("All 200 customers, coloured by group:")

        st.scatter_chart(
            customers,
            x="Annual Income (k$)",
            y="Spending Score (1-100)",
            color="group_name"
        )


# ==========================================
# TAB 3: ABOUT
# ==========================================

with tab_about:

    st.subheader("1. Fixed deposit model (supervised)")

    st.write(
        "A Random Forest classification model predicts "
        "whether a customer may open a fixed deposit."
    )

    st.write(
        f"Best model: **{savings['model_name']}**"
    )

    st.dataframe(
        pd.DataFrame(savings["scores"]),
        hide_index=True
    )

    st.caption(
        "For the original workshop, replace the generated "
        "demonstration data with the Kaggle Bank Marketing dataset."
    )

    st.divider()

    st.subheader("2. Saver-group model (unsupervised)")

    st.write(
        "K-Means identifies five customer groups using "
        "annual income and spending score."
    )

    st.dataframe(
        customers["group_name"].value_counts().rename("Customers")
    )

    st.caption(
        "For the original workshop, replace the generated "
        "demonstration data with the Kaggle Mall Customer "
        "Segmentation dataset."
    )

    st.divider()

    st.subheader("3. Model information")

    st.write(
        "- Classification: Random Forest\n"
        "- Clustering: K-Means\n"
        "- Preprocessing: One-Hot Encoding and StandardScaler\n"
        "- Model storage: Joblib"
    )

    st.warning(
        "This version uses synthetic demonstration data, "
        "not the original Kaggle training datasets. "
        "Its predictions are illustrative only."
    )
