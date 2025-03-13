"""
Database module for handling data storage in both PostgreSQL and Supabase.
"""

# Use a try/except approach to support both running from app directory and project root
try:
    # Try importing with app prefix (for running from project root)
    from app.modules.db.utils import (
        save_raw_data_redundant,
        save_enriched_data_redundant,
        save_preprocessed_data_redundant
    )

    from app.modules.db.data_add import (
        add_raw_data_to_db,
        add_enriched_data_to_db,
        add_preprocessed_data_to_db,
        get_latest_listings
    )

    from app.modules.db.new_data import (
        start_file_watcher,
        DataFileHandler
    )
except ImportError:
    # Fall back to direct import (for running from app directory)
    from modules.db.utils import (
        save_raw_data_redundant,
        save_enriched_data_redundant,
        save_preprocessed_data_redundant
    )

    from modules.db.data_add import (
        add_raw_data_to_db,
        add_enriched_data_to_db,
        add_preprocessed_data_to_db,
        get_latest_listings
    )

    from modules.db.new_data import (
        start_file_watcher,
        DataFileHandler
    )

# Make these easily importable
__all__ = [
    # Redundant storage functions
    'save_raw_data_redundant',
    'save_enriched_data_redundant',
    'save_preprocessed_data_redundant',
    
    # Data addition functions
    'add_raw_data_to_db',
    'add_enriched_data_to_db',
    'add_preprocessed_data_to_db',
    'get_latest_listings',
    
    # File watcher
    'start_file_watcher',
    'DataFileHandler'
] 