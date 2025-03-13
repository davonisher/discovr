import os
import sys

# Global API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
brevo_api_key = os.getenv("BREVO_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
deepseek_api_key = os.getenv("DEEPSEEK_API_KEY")

# Setup directory paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # Go up one level from app directory

# Define the directory to store saved CSV files
SAVED_RUNS_DIR = os.path.join(project_root, "data", "raw_data")
os.makedirs(SAVED_RUNS_DIR, exist_ok=True)

SAVED_RUNS_DIR_PP = os.path.join(project_root, "data", "pp")
os.makedirs(SAVED_RUNS_DIR_PP, exist_ok=True)

PP = os.path.join(project_root, "data", "pp_data") 
os.makedirs(PP, exist_ok=True)

# File paths
USER_AGENTS_PATH = os.path.join(current_dir, "data", "user-agents.txt")
CAT_FILE_PATH = os.path.join(current_dir, "data", "cat.txt")
CSV_EXPORT_PATH = os.path.join(project_root, "data", "class_data", "{search_query_encoded}.csv")

# Add project root to path for module imports
if project_root not in sys.path:
    sys.path.append(project_root) 

# Database configuration
database_url = os.getenv("DATABASE_URL")
if not database_url:
    raise ValueError("DATABASE_URL environment variable must be set")

# Supabase configuration
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if not supabase_url or not supabase_key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY environment variables must be set")
