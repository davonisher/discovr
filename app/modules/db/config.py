"""
Database configuration settings for both PostgreSQL and Supabase.
"""

# PostgreSQL connection
# Use environment variables instead of hardcoded credentials
import os

# Get database connection parameters from environment variables
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', 'katse')
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'postgres')


# Supabase connection
supabase_url = "https://skyehzreswvdtbfuhupa.supabase.co"
supabase_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNreWVoenJlc3d2ZHRiZnVodXBhIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDAzOTAwNjMsImV4cCI6MjA1NTk2NjA2M30.N9Hb72TxomkeFCXy4wZPXElSvKB2l5SHbHFN6yJgyFY" 

