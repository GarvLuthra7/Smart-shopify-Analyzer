import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load models
@st.cache_resource
def load_models():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model_s, model_r = load_models()

st.set_page_config(page_title="Smart Shopping Assistant Pro", layout="wide")

st.title("🛒 Smart Shopping Decision Assistant")
st.markdown("---")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.header("📥 Enter Details")

    price = st.number_input("Product Price (₹)", min_value=100, value=5000)
    salary = st.number_input("Monthly Salary (₹)", min_value=5000, value=50000)
    expenses = st.number_input("Monthly Expenses (₹)", min_value=1000, value=30000)

    usage = st.slider("Usage Frequency (times/month)", 1, 30, 10)

    product_type = st.selectbox(
        "Product Category",
        ["Essential", "Productivity", "Lifestyle", "Luxury"]
    )

    product_type_score_map = {"Essential": 1.0, "Productivity": 0.8, "Lifestyle": 0.5, "Luxury": 0.2}
    product_type_score = product_type_score_map[product_type]

    st.subheader("🧠 Need Assessment")
    q1 = st.checkbox("Improves productivity?")
    q2 = st.checkbox("Cannot delay 30 days?")
    q3 = st.checkbox("Replacing essential item?")
    q4 = st.checkbox("Will use frequently?")
    q5 = st.checkbox("Required for work/study/health?")

    need_score = (q1 + q2 + q3 + q4 + q5) / 5
    final_need_score = (need_score * 0.7) + (product_type_score * 0.3)

    st.write(f"📊 Need Score: **{round(final_need_score, 2)}**")
    analyze = st.button("🔍 Analyze Purchase", use_container_width=True)

with col2:
    st.header("📊 Results")

    if analyze:
        # Current State Data
        input_data = pd.DataFrame([[price, salary, expenses, usage, final_need_score]], 
                        columns=["price", "salary", "expenses", "usage", "need_score"])
        #input_data = np.array([[price, salary, expenses, usage, final_need_score]])
        
        # Original Prediction
        safety = model_s.predict(input_data)[0]
        regret = model_r.predict(input_data)[0]
        safety = max(0, min(1, safety))
        regret = max(0, min(1, regret))

        # 1. VISUALIZE COST-PER-USE (Utility Analysis)
        cost_per_use = price / usage
        st.subheader("💡 Utility Value")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("Financial Safety", f"{int(safety * 100)}%")
        m_c2.metric("Cost Per Use", f"₹{round(cost_per_use, 2)}")
        
        st.info(f"**Insight:** This item costs you ₹{round(cost_per_use, 2)} every time you use it this month.")
        st.divider()

        # 2. FEATURE INTERPRETABILITY (The ML Angle)
        st.subheader("🔍 Why this score?")
        discretionary_income = salary - expenses
        price_impact = (price / (discretionary_income + 1)) * 100
        
        feature_names = ['Price Impact', 'Need Score', 'Usage Utility']
        impact_values = [price_impact, final_need_score * 10, usage / 3] 
        chart_data = pd.DataFrame({"Factor": feature_names, "Impact Strength": impact_values})
        
        st.bar_chart(chart_data.set_index("Factor"))
        st.caption("Lower 'Price Impact' and higher 'Need/Usage' lead to better safety scores.")

        st.divider()
        st.subheader("🛠️ Prescriptive Action")
        
        if safety > 0.6:
            st.success("🟢 Buy Now – High financial stability.")
        else:
            delay_days = int((1 - safety) * 15) if safety < 0.35 else int((1 - safety) * 7)
            delay_days = max(3, delay_days)
            
            simulated_data = np.array([[price, salary, expenses * 0.8, usage, final_need_score]])
            simulated_safety = model_s.predict(simulated_data)[0]
            simulated_safety = max(0, min(1, simulated_safety))
            
            st.error(f"🔴 Avoid Purchase" if safety < 0.35 else "🟡 Think Twice")
            
            st.markdown(f"""
            > **Prescriptive Note:** Delaying this purchase by **{delay_days} days** (allowing your monthly buffer to reset) 
            > would increase your Financial Safety from **{int(safety*100)}%** to **{int(simulated_safety*100)}%**.
            """)
