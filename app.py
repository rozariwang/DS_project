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

st.title("🍎 Data Science Group 3 🚀 US Stock Insights: Smart Analysis for Long-Term Growth")

st.text('Members: Kate Rebecca Belcher, Nicholas Jennings, William LaCroix, Myeongju Lee, Ho-Hsuan Wang (in alphabetical order)')

# Introduction section
st.markdown(
    """
    Welcome to the Long-Term Investment Analysis App! 📊 This app helps investors make informed decisions for
    long-term investment success. By leveraging advanced large language models (LLMs), analyzing macroeconomic
    indicators, and monitoring industry trends, it provides insights into the growth potential of stocks.
    
    Please note that investing in the stock market carries risks, and past performance is not always indicative of
    future results. The Long-Term Investment Analysis App does not guarantee investment success or provide
    financial advice. It is essential to evaluate your investment decisions carefully and consider consulting
    with a qualified financial professional. 
    
    Happy Investing! 💸 
    """
)

with open('company_names.txt', 'r') as file:
    companies= file.readlines() 
    
# Initialize search history list 
search_history = []
    
# Streamlit app code
def main():
    st.subheader("Investment Recommendation")
    
    # Dropdown selection box for companies
    selected_company = st.selectbox("What company do you want to invest in?", companies)
    
    # Submit button
    submit_button = st.button("Generate Recommendation")
    
    if submit_button:
        if selected_company:
            # Show loading message
            with st.spinner("Generating recommendation..."):
                # Call controller's generate_recommendation() function to get recommendation
                response = controller.generate_recommendation(selected_company)
                
                # Display the recommendation
                st.write(response)
                # Append the search history with a maximum of three entries
                if len(search_history) == 3:
                    search_history.pop(0)  # Remove the oldest entry if the search history is full
                search_history.append((selected_company, response))
   
    # Store updated search history in cache
    get_search_history(search_history)
    
    # Display search history section
    st.subheader("Search History")
    for company, recommendation in search_history:
        st.write(f"Company: {company}")
        st.write(f"Recommendation: {recommendation}")
        st.divider()
        
if __name__ == '__main__':
    main()
    