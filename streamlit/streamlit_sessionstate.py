"""
Example code to test session state for streamlit.

"""
import SessionState # Download from https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
import streamlit as st


# Initialize session state and variables
ss = SessionState.get(num0=0)
num1 = 0

# Increment test
if st.button('Increment'):
    ss.num0 += 1
    num1 += 1

# Print result
st.write('With SessionState: %d' % ss.num0)
st.write('Without SessionState: %d' % num1)
