import controller
import streamlit as st

st.image("saarlogo.png", use_column_width=True)

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

st.title("ProfitProphet")
st.markdown("🍎 Data Science Group 3 🚀 US Stock Insights: Smart Analysis for Long-Term Growth 🔮")

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

@st.cache_data
def load_companies():
    """
    Loads the list of companies from the text file.
    Returns:
        list: A list of companies.
    """
    with open('stock_names_with_tickers.txt', 'r', encoding='utf-8') as file_in:
        companies = file_in.read().splitlines()
    return companies

# Streamlit app code
def main():
    """
    Main function of the app.
    Allows company selection and displays the recommendation.
    args:
        None.
    Returns:
        None.
    """
    st.subheader("Investment Recommendation")

    # Dropdown selection box for companies
    #selected_company = st.selectbox("##### What company do you want to invest in?", load_companies())
    st.markdown("<h4 style='text-align: center;'>What company do you want to invest in?</h4>", unsafe_allow_html=True)
    st.empty()
    selected_company = st.selectbox("", load_companies())
    st.markdown("<p style='font-size: 12px; margin-top: -15px; text-align: center;'>You can either select a company from the list below or simply type in your selection.</p>", unsafe_allow_html=True)
    
    #align button in the centre
    container = st.container()
    with container:
        st.write("")
        st.markdown("<p style='text-align: center;'>Regrettably, our app's predictions are limited to specific companies on Nasdaq and the New York Stock Exchange at the moment. We apologize for any inconvenience if the company you are interested in is not covered.</p>", unsafe_allow_html=True)
        st.write("")

    centered_button = container.empty()
    with centered_button:
        st.button("Generate Recommendation")
    
    if submit_button:
        if selected_company:
            # Show loading message
            with st.spinner("Generating recommendation..."):
                # Call controller's generate_recommendation() function to get recommendation
                response = controller.generate_recommendation(selected_company)
                if response == 404:
                    st.error("Sorry, we could not find the company you selected. Please try again.")
                    return
                # Display the recommendation
                st.divider()
                st.write(response)

if __name__ == '__main__':
    main()
    
