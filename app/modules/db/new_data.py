import os
import sys
import time
import logging
from typing import Set
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import pandas as pd

# Fix the path to properly import from parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(parent_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from modules.db.data_add import add_raw_data_to_db, add_enriched_data_to_db, add_preprocessed_data_to_db
from modules.db.ps.database import create_tables_if_not_exists as create_postgres_tables
from modules.db.utils import save_preprocessed_data_redundant

class DataFileHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        self.processed_files: Set[str] = set()
        
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            self._process_file(event.src_path)
            
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.csv'):
            self._process_file(event.src_path)
            
    def _process_file(self, file_path: str):
        """Process a new or modified CSV file."""
        if file_path in self.processed_files:
            return
            
        try:
            # Add file to processed set
            self.processed_files.add(file_path)
            
            # Log the new file
            filename = os.path.basename(file_path)
            logging.info(f"Processing new file: {filename}")
            
            # Wait a short time to ensure file is completely written
            time.sleep(1)
            
            # Process raw data
            if "Raw_" in filename:
                logging.info(f"Detected raw data file: {filename}")
                success = add_raw_data_to_db(file_path)
                if success:
                    logging.info(f"Successfully processed raw data file: {filename}")
                else:
                    logging.error(f"Failed to process raw data file: {filename}")
                    
            # Process preprocessed data
            elif "Preprocessed_" in filename:
                logging.info(f"Detected preprocessed data file: {filename}")
                # Load CSV and store preprocessed data into database
                df = pd.read_csv(file_path)
                success = add_preprocessed_data_to_db(df, source_file=filename)
                if success:
                    logging.info(f"Successfully processed preprocessed data file: {filename}")
                else:
                    logging.error(f"Failed to process preprocessed data file: {filename}")
                    
            # Process enriched data
            elif "Enriched_" in filename:
                logging.info(f"Detected enriched data file: {filename}")
                success = add_enriched_data_to_db(file_path)
                if success:
                    logging.info(f"Successfully processed enriched data file: {filename}")
                else:
                    logging.error(f"Failed to process enriched data file: {filename}")
            else:
                logging.info(f"Skipping file with unknown format: {filename}")
                    
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")

def start_file_watcher():
    """
    Start watching the raw_data directory for new CSV files.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(project_root, 'data_processing.log'))
        ]
    )
    
    # Ensure database tables exist
    create_postgres_tables()
    
    # Set up the file watcher
    raw_data_path = os.path.join(project_root, "data", "raw_data")
    event_handler = DataFileHandler()
    observer = Observer()
    
    try:
        # Start watching the directory
        observer.schedule(event_handler, raw_data_path, recursive=False)
        observer.start()
        logging.info(f"Started watching directory: {raw_data_path}")
        
        # Process any existing files that haven't been processed
        for filename in os.listdir(raw_data_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(raw_data_path, filename)
                event_handler._process_file(file_path)
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
            logging.info("File watcher stopped by user")
            
        observer.join()
        
    except Exception as e:
        logging.error(f"Error in file watcher: {e}")
        observer.stop()
        observer.join()

if __name__ == "__main__":
    start_file_watcher()
