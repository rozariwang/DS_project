import streamlit as st
import openai
import controller

st.title("🍎 Data Science Group 3 - NLP Financial Forcasting App")

# Define the list of tech companies
companies = [
    "Apple",
    "Amazon",
    "Google",
    "Microsoft"
    "Facebook",
    "Tesla",
    # Add more companies as desired
]

# Streamlit app code
def main():
    st.title("Tech Company Investment Recommendation")
    
    # Dropdown selection box for companies
    selected_company = st.selectbox("What company do you want to invest in?", companies)

    if selected_company:
        # Call controller's generate_recommendation() function to get recommendation
        response = controller.generate_recommendation(selected_company)
        
        # Display recommendation
        st.subheader("Recommendation:")
        st.write(response)

if __name__ == '__main__':
    main()