
import streamlit as st
import os
from supabase import create_client
from app.config import supabase_url, supabase_key

def login_page():
    """
    Display a login page with options to login using Streamlit's built-in authentication
    or using Supabase authentication.
    """
    st.title("🔐 Login to Discovr")
    
    login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
    
    with login_tab:
        st.subheader("Login to your account")
        
        # Option 1: Streamlit's built-in authentication
        st.write("### Option 1: Login with your provider")
        if st.button("Login with Google"):
            st.login()
            
        # Option 2: Email/Password login with Supabase
        st.write("### Option 2: Login with email and password")
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            if email and password:
                try:
                    # Initialize Supabase client
                    supabase = create_client(supabase_url, supabase_key)
                    
                    # Authenticate user
                    response = supabase.auth.sign_in_with_password({"email": email, "password": password})
                    
                    if response.user:
                        st.session_state.user = response.user
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Login failed. Please check your credentials.")
                except Exception as e:
                    st.error(f"Login error: {str(e)}")
            else:
                st.warning("Please enter both email and password.")
    
    with signup_tab:
        st.subheader("Create a new account")
        
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up"):
            if new_email and new_password and confirm_password:
                if new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    try:
                        # Initialize Supabase client
                        supabase = create_client(supabase_url, supabase_key)
                        
                        # Create new user
                        response = supabase.auth.sign_up({
                            "email": new_email,
                            "password": new_password
                        })
                        
                        if response.user:
                            st.success("Account created successfully! Please check your email to verify your account.")
                        else:
                            st.error("Failed to create account.")
                    except Exception as e:
                        st.error(f"Sign up error: {str(e)}")
            else:
                st.warning("Please fill in all fields.")
    
    # Logout button if user is already logged in
    if st.experimental_user.is_logged_in or 'user' in st.session_state:
        if st.button("Logout"):
            if 'user' in st.session_state:
                del st.session_state.user
            st.logout()
            st.rerun()
