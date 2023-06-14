import streamlit as st
import controller

st.title("🍎 Data Science Group 3 - NLP Financial Forcasting App")

with open('company_names.txt', 'r') as file:
    companies= file.readlines() 
    
# Streamlit app code
def main():
    st.title("💸 US Company Investment Recommendation 🚀")
    
    # Dropdown selection box for companies
    selected_company = st.selectbox("What company do you want to invest in?", companies)
    
    # Empty placeholder for apology message
    apology_placeholder = st.empty()
    
    # Submit button
    submit_button = st.button("Generate Recommendation")
    
    if submit_button:
        if selected_company:
        # Call controller's generate_recommendation() function to get recommendation
          response = controller.generate_recommendation(selected_company)
          
          if response is None:
              apology_placeholder.subheader("Apologies!")
              apology_placeholder.write(
                    f"We cannot provide a recommendation for stock investment in company {selected_company}, "
                    "as there is no data available for this company."
               )
          else:
              # Display recommendation
              st.subheader("Recommendation:")
              st.write(response)

if __name__ == '__main__':
    main()
    