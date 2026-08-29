import streamlit as st
import pandas as pd
import joblib


# ===========================
# Page Configuration
# ===========================

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered"
)


# ===========================
# Load Model and Processed Data
# ===========================

model = joblib.load("salary_model.pkl")

processed_df = pd.read_csv("salary_processed.csv")


# ===========================
# Create Job Title Mapping
# ===========================

job_titles = {}

for _, row in processed_df.iterrows():

    # The processed dataset contains encoded Job Title values.
    # If your processed dataset has only encoded values,
    # we cannot recover the original job names from it.
    pass


# ===========================
# Job Titles
# ===========================

job_titles = [
    "Account Manager",
    "Accountant",
    "Administrative Assistant",
    "Business Analyst",
    "CEO",
    "Customer Service Manager",
    "Customer Service Rep",
    "Customer Success Rep",
    "Data Analyst",
    "Data Entry Clerk",
    "Data Scientist",
    "Director",
    "Director of Marketing",
    "Financial Analyst",
    "Financial Manager",
    "Help Desk Analyst",
    "HR Generalist",
    "HR Manager",
    "IT Support",
    "Junior Developer",
    "Marketing Analyst",
    "Marketing Coordinator",
    "Marketing Manager",
    "Marketing Specialist",
    "Network Engineer",
    "Operations Director",
    "Operations Manager",
    "Product Designer",
    "Product Engineer",
    "Product Manager",
    "Project Engineer",
    "Project Manager",
    "Recruiter",
    "Sales Associate",
    "Sales Director",
    "Sales Executive",
    "Sales Manager",
    "Senior Consultant",
    "Senior Engineer",
    "Senior Manager",
    "Senior Scientist",
    "Social Media Specialist",
    "Software Developer",
    "Software Engineer",
    "Software Manager",
    "Strategy Consultant",
    "Technical Writer",
    "UX Designer",
    "VP of Operations"
]


# ===========================
# IMPORTANT
# Job Title Encoding
# ===========================

# Use the encoding from your processed dataset.
# This assumes Job Title was encoded alphabetically
# using LabelEncoder.

from sklearn.preprocessing import LabelEncoder

# Load original job titles from the processed dataset
# If encoded values are already present, create a mapping
# based on the job title list.

job_encoder = LabelEncoder()

job_encoder.fit(job_titles)

job_mapping = {
    title: int(job_encoder.transform([title])[0])
    for title in job_titles
}


# ===========================
# Education Mapping
# ===========================

education_mapping = {
    "Bachelor's": 1,
    "Master's": 2,
    "PhD": 3
}


# ===========================
# Gender Mapping
# ===========================

gender_mapping = {
    "Female": 0,
    "Male": 1
}


# ===========================
# Custom CSS
# ===========================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 30px;
    }

    .salary-box {
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
        border: 1px solid rgba(128, 128, 128, 0.3);
    }

    .salary-label {
        font-size: 18px;
        font-weight: 500;
    }

    .salary-value {
        font-size: 38px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ===========================
# Header
# ===========================

st.markdown(
    '<div class="main-title">💼 Employee Salary Predictor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict an estimated annual salary using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ===========================
# Employee Information
# ===========================

st.subheader("👤 Employee Information")

col1, col2 = st.columns(2)


with col1:

    age = st.number_input(
        "🎂 Age",
        min_value=18,
        max_value=70,
        value=30,
        step=1
    )


with col2:

    gender = st.selectbox(
        "⚧ Gender",
        ["Male", "Female"]
    )


# ===========================
# Education and Experience
# ===========================

col1, col2 = st.columns(2)


with col1:

    education = st.selectbox(
        "🎓 Education Level",
        [
            "Bachelor's",
            "Master's",
            "PhD"
        ]
    )


with col2:

    experience = st.number_input(
        "💼 Years of Experience",
        min_value=0.0,
        max_value=50.0,
        value=5.0,
        step=0.5
    )


# ===========================
# Job Title
# ===========================

job_title = st.selectbox(
    "🏢 Job Title",
    sorted(job_titles)
)


st.divider()


# ===========================
# Prediction
# ===========================

if st.button(
    "💰 Predict Salary",
    use_container_width=True
):

    # Convert user input to encoded values

    employee = {

        "Age": age,

        "Education Level":
            education_mapping[education],

        "Job Title":
            job_mapping[job_title],

        "Years of Experience":
            experience,

        "Gender":
            gender_mapping[gender]
    }


    # Create DataFrame

    employee_df = pd.DataFrame(
        [employee]
    )


    # Prediction

    prediction = model.predict(
        employee_df
    )[0]


    # ===========================
    # Display Result
    # ===========================

    st.success(
        "Prediction generated successfully!"
    )


    st.markdown(
        '<div class="salary-box">',
        unsafe_allow_html=True
    )


    st.markdown(
        '<div class="salary-label">'
        'Estimated Annual Salary'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        f'<div class="salary-value">'
        f'₹{prediction:,.0f}'
        f'</div>',
        unsafe_allow_html=True
    )


    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


    # ===========================
    # Prediction Details
    # ===========================

    st.subheader(
        "📋 Prediction Details"
    )


    col1, col2 = st.columns(2)


    with col1:

        st.write(
            f"**Age:** {age}"
        )

        st.write(
            f"**Gender:** {gender}"
        )

        st.write(
            f"**Education:** {education}"
        )


    with col2:

        st.write(
            f"**Job Title:** {job_title}"
        )

        st.write(
            f"**Experience:** "
            f"{experience} years"
        )


    st.info(
        "ℹ️ This prediction is an estimate "
        "based on patterns learned from the "
        "training dataset."
    )