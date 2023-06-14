import streamlit as st
import controller

# Add custom CSS to align content in the middle
st.markdown(
    """
    <style>
    .stMarkdown, .stTitle {
        display: flex;
        justify-content: center;
        align-items: center;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🍎 Data Science Group 3 - NLP Financial Forcasting App")

# Introduction section
st.markdown(
    """
    ## Introduction
    
    Welcome to the NLP Financial Forecasting App! This app provides investment recommendations
    for US companies based on natural language processing techniques and financial data analysis.
    
    Simply select a company from the dropdown list, click the Generate Recommendation button, and receive a 
    recommendation tailored to your investment needs.
    
    Happy investing!
    """
)

with open('company_names.txt', 'r') as file:
    companies= file.readlines() 
    
# Streamlit app code
def main():
    st.title("💸 US Company Investment Recommendation 🚀")
    
    # Dropdown selection box for companies
    selected_company = st.selectbox("What company do you want to invest in?", companies)
    
    # Submit button
    submit_button = st.button("Generate Recommendation")
    
    if submit_button:
        if selected_company:
        # Call controller's generate_recommendation() function to get recommendation
          response = controller.generate_recommendation(selected_company)

if __name__ == '__main__':
    main()
    