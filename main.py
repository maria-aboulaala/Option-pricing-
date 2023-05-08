import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from datetime import datetime, timedelta
from scipy.stats import norm
st.set_page_config(page_icon=":game_die:", page_title="Aboulaala Projet")

st.image('WhatsApp_Image_2022-06-10_at_23.30.07-removebg-preview.png',caption=None, width=250, use_column_width=None, clamp = False, channels="RGB", output_format="auto")
st.title('Projet Pricing des options :bar_chart:')

with st.expander("Presentation"):

    st.markdown(
    """
> Cette presentation est faite dans le cadre du projet Evaluation des options europeennes par la methode de MonteCarlo dans le modele de Black-Scholes
- Realiser par : Aboulaala Maria 
- Encadrer par : Pr. Brahim El Asri
   """
)

st.markdown(""" 
Les parametres demander par l'utilisateur sont: \n
>S: est le prix actuel de l actif sous jacent (par exemple une action \n
>K:  est le prix d'exercice de l'option \n
>r: est le taux d'intérêt sans risque \n
>sigma: volatilite \n
>T: est la durée restante de l'option ; \n
>n_simulations: number of simulations   \n
                
                """)

st.subheader('Entrez les parametres: :key: ')


with st.form(key="my_form"):
    T = st.number_input('la durée restante de l option', step= 1)   
    date = st.date_input('La date')
    stock_name = st.selectbox(
    'Le symbole de stock',
    ('AAPL', 'MSFT', 'META', 'GOOG', 'AMZN'))
    nSim = st.number_input('Le nombre de simulation', step=1, min_value=1)
    K = st.number_input('le prix d exercice de l option')
    r = st.number_input('le taux d intérêt sans risque', min_value=0, max_value=1)
    
    st.form_submit_button("Simuler")









def black_scholes_monte_carlo_call(S, K, r, sigma, T, n_simulations):
    """
    Calculate the price of a European call option using the Black-Scholes-Merton model and Monte Carlo simulation.
    
    S: spot price of the underlying asset
    K: strike price of the option
    r: risk-free interest rate
    sigma: volatility of the underlying asset
    T: time to expiration (in years)
    n_simulations: number of simulations to run
    """
    dt = T / 365.0
    S_t = np.zeros(n_simulations)
    S_t[...] = S
    for i in range(int(T / dt)):
        epsilon = np.random.normal(size=n_simulations)
        S_t *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)
    call_price = np.exp(-r * T) * np.maximum(S_t - K, 0.0).mean()
    return call_price

def black_scholes_monte_carlo_put(S, K, r, sigma, T, n_simulations):
    """
    Calculate the price of a European put option using the Black-Scholes-Merton model and Monte Carlo simulation.
    
    S: spot price of the underlying asset
    K: strike price of the option
    r: risk-free interest rate
    sigma: volatility of the underlying asset
    T: time to expiration (in years)
    n_simulations: number of simulations to run
    """
    dt = T / 365.0
    S_t = np.zeros(n_simulations)
    S_t[...] = S
    for i in range(int(T / dt)):
        epsilon = np.random.normal(size=n_simulations)
        S_t *= np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * epsilon)
    put_price = np.exp(-r * T) * np.maximum(K - S_t, 0.0).mean()
    return put_price

# Example usage
S = 100.0  # Spot price of the underlying asset
K = 105.0  # Strike price of the option
r = 0.05   # Risk-free interest rate
sigma = 0.20  # Volatility of the underlying asset
#T = 1.0    # Time to expiration (in years)
n_simulations = 1000000  # Number of Monte Carlo simulations to run

call_price = black_scholes_monte_carlo_call(S, K, r, sigma, T, n_simulations)
put_price = black_scholes_monte_carlo_put(S, K, r, sigma, T, n_simulations)

st.subheader('Resultats:key: ')
st.write('la valeur du call est',call_price)
st.write('la valeur du put est',put_price)

