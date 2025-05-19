# 8. Technical Implementation Details

# ---- Project Setup ----

# requirements.txt
"""
# Data Processing
numpy==1.24.3
pandas==2.0.2
astropy==5.3.1
scipy==1.10.1

# Machine Learning
scikit-learn==1.2.2
tensorflow==2.12.0
keras==2.12.0

# Visualization
matplotlib==3.7.1
seaborn==0.12.2
plotly==5.14.1
dash==2.10.2

# Interactive Development
jupyter==1.0.0
ipywidgets==8.0.6

# Documentation and Testing
sphinx==6.1.3
pytest==7.3.1

# Utility
tqdm==4.65.0
"""

# ---- Docker Configuration ----

# Dockerfile
"""
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create non-root user
RUN useradd -m exouser
USER exouser

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
"""

# ---- Project Structure ----

# project_structure.py
def create_project_structure():
    """
    Creates the directory structure for the exoplanet habitability analysis project.
    """
    import os
    
    directories = [
        "data/raw",
        "data/processed",
        "data/interim",
        "notebooks",
        "src/data",
        "src/features",
        "src/models",
        "src/visualization",
        "src/utils",
        "reports/figures",
        "reports/documents",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, ".gitkeep"), "w") as f:
            pass
    
    print("Project structure created successfully!")

# Example usage
if __name__ == "__main__":
    create_project_structure()

# ---- Data Processing Pipeline ----

# src/data/data_pipeline.py
import pandas as pd
import numpy as np
from astropy.io import fits
import os
from typing import Dict, List, Optional, Union

class ExoplanetDataPipeline:
    """
    Pipeline for processing exoplanet data from various sources.
    """
    
    def __init__(self, config_path: str = "config/data_config.json"):
        """
        Initialize the data pipeline with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        import json
        
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.raw_data_dir = self.config.get("raw_data_dir", "data/raw")
        self.processed_data_dir = self.config.get("processed_data_dir", "data/processed")
        self.interim_data_dir = self.config.get("interim_data_dir", "data/interim")
    
    def load_nasa_exoplanet_archive(self) -> pd.DataFrame:
        """
        Load data from NASA Exoplanet Archive CSV file.
        
        Returns:
            DataFrame containing exoplanet data
        """
        file_path = os.path.join(self.raw_data_dir, "nasa_exoplanet_archive.csv")
        return pd.read_csv(file_path)
    
    def load_kepler_data(self) -> pd.DataFrame:
        """
        Load Kepler mission data from FITS files.
        
        Returns:
            DataFrame containing Kepler data
        """
        file_path = os.path.join(self.raw_data_dir, "kepler_data.fits")
        with fits.open(file_path) as hdul:
            data = hdul[1].data
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(data)
        return df
    
    def clean_exoplanet_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess exoplanet data.
        
        Args:
            df: Raw exoplanet data
            
        Returns:
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        
        # Fill numerical missing values with median
        for col in numerical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Fill categorical missing values with most frequent value
        for col in categorical_cols:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Save intermediate result
        interim_path = os.path.join(self.interim_data_dir, "cleaned_exoplanet_data.csv")
        df_clean.to_csv(interim_path, index=False)
        
        return df_clean
    
    def calculate_habitable_zone_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate features related to habitable zone analysis.
        
        Args:
            df: Cleaned exoplanet data
            
        Returns:
            DataFrame with additional features
        """
        df_features = df.copy()
        
        # Calculate distance from host star in AU
        if 'semi_major_axis_au' not in df_features.columns:
            if 'period_days' in df_features.columns and 'star_mass' in df_features.columns:
                # Use Kepler's 3rd law to estimate semi-major axis
                # a^3 = (P^2 * G * M) / (4 * π^2)
                # Simplified for period in days, mass in solar masses, result in AU
                df_features['semi_major_axis_au'] = ((df_features['period_days'] / 365.25) ** 2 * 
                                                   df_features['star_mass']) ** (1/3)
        
        # Calculate equilibrium temperature if not present
        if 'equilibrium_temperature' not in df_features.columns:
            if 'star_effective_temp' in df_features.columns and 'semi_major_axis_au' in df_features.columns:
                # Simple equilibrium temperature model: T_eq = T_star * sqrt(R_star / (2 * a))
                df_features['equilibrium_temperature'] = df_features['star_effective_temp'] * \
                    np.sqrt(df_features['star_radius'] / (2 * df_features['semi_major_axis_au'] * 215))
        
        # Calculate habitable zone flag
        df_features['in_habitable_zone'] = False
        
        # Basic habitable zone criterion (adjust based on your model)
        mask = ((df_features['equilibrium_temperature'] >= 180) & 
                (df_features['equilibrium_temperature'] <= 310))
        df_features.loc[mask, 'in_habitable_zone'] = True
        
        # Save processed data
        processed_path = os.path.join(self.processed_data_dir, "exoplanet_data_with_features.csv")
        df_features.to_csv(processed_path, index=False)
        
        return df_features
    
    def run_pipeline(self) -> pd.DataFrame:
        """
        Run the complete data processing pipeline.
        
        Returns:
            Fully processed DataFrame ready for analysis
        """
        # Load data
        try:
            df_nasa = self.load_nasa_exoplanet_archive()
            print(f"Loaded {len(df_nasa)} records from NASA Exoplanet Archive")
        except Exception as e:
            print(f"Error loading NASA data: {e}")
            df_nasa = pd.DataFrame()
        
        try:
            df_kepler = self.load_kepler_data()
            print(f"Loaded {len(df_kepler)} records from Kepler data")
        except Exception as e:
            print(f"Error loading Kepler data: {e}")
            df_kepler = pd.DataFrame()
        
        # Merge datasets if both are available
        if not df_nasa.empty and not df_kepler.empty:
            # This requires defining a proper merging strategy based on common columns
            # For example:
            common_cols = list(set(df_nasa.columns) & set(df_kepler.columns))
            if common_cols:
                df_combined = pd.merge(df_nasa, df_kepler, on=common_cols, how='outer')
            else:
                print("Warning: No common columns for merging. Using NASA data only.")
                df_combined = df_nasa
        elif not df_nasa.empty:
            df_combined = df_nasa
        elif not df_kepler.empty:
            df_combined = df_kepler
        else:
            raise ValueError("No data available to process")
        
        # Clean data
        df_clean = self.clean_exoplanet_data(df_combined)
        
        # Calculate features
        df_features = self.calculate_habitable_zone_features(df_clean)
        
        print(f"Pipeline completed successfully. Final dataset has {len(df_features)} records.")
        return df_features

# ---- Version Control Setup ----

# setup_git.py
def setup_git_repository():
    """
    Initialize git repository and create .gitignore file.
    """
    import subprocess
    import os
    
    # Initialize git repository
    subprocess.run(["git", "init"])
    
    # Create .gitignore file
    gitignore_content = """
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Distribution / packaging
dist/
build/
*.egg-info/

# Unit test / coverage reports
htmlcov/
.coverage
.coverage.*

# Jupyter Notebook
.ipynb_checkpoints

# Environment
.env
.venv
env/
venv/
ENV/

# Data (large files should be kept out of git)
data/raw/*
data/processed/*
data/interim/*
!data/raw/.gitkeep
!data/processed/.gitkeep
!data/interim/.gitkeep

# IDE specific files
.idea/
.vscode/
*.swp
*.swo

# OS specific files
.DS_Store
Thumbs.db
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    # Initial commit
    subprocess.run(["git", "add", "."])
    subprocess.run(["git", "commit", "-m", "Initial commit: Project structure and setup"])
    
    print("Git repository initialized successfully!")

# ---- Logging Configuration ----

# src/utils/logger.py
import logging
import os
from datetime import datetime

def setup_logger(name: str, log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger with specified name and level.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Create logs directory if it doesn't exist
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if log_file is specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# Example usage
project_logger = setup_logger(
    name="exoplanet_project",
    log_file="logs/exoplanet_project.log",
    level=logging.INFO
)

# ---- Data Security and Backup ----

# src/utils/backup.py
import shutil
import os
import datetime
import tarfile
import glob

def create_backup(source_dir: str, backup_dir: str) -> str:
    """
    Create a backup of the specified directory.
    
    Args:
        source_dir: Directory to backup
        backup_dir: Directory to store the backup
    
    Returns:
        Path to the created backup file
    """
    # Create backup directory if it doesn't exist
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    
    # Generate backup filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_filename = f"backup_{os.path.basename(source_dir)}_{timestamp}.tar.gz"
    backup_path = os.path.join(backup_dir, backup_filename)
    
    # Create tarball
    with tarfile.open(backup_path, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
    
    print(f"Backup created: {backup_path}")
    return backup_path

def rotate_backups(backup_dir: str, max_backups: int = 5) -> None:
    """
    Remove old backups, keeping only the specified number of most recent backups.
    
    Args:
        backup_dir: Directory containing backups
        max_backups: Maximum number of backups to keep
    """
    # List all backup files
    backup_files = glob.glob(os.path.join(backup_dir, "backup_*.tar.gz"))
    
    # Sort by modification time (newest first)
    backup_files.sort(key=os.path.getmtime, reverse=True)
    
    # Remove old backups
    for old_backup in backup_files[max_backups:]:
        os.remove(old_backup)
        print(f"Removed old backup: {old_backup}")

# Example scheduled backup script
def scheduled_backup():
    """
    Perform a scheduled backup of important project directories.
    """
    # Directories to backup
    dirs_to_backup = [
        "data/processed",
        "data/interim",
        "notebooks",
        "src",
        "reports"
    ]
    
    backup_dir = "backups"
    
    for directory in dirs_to_backup:
        if os.path.exists(directory):
            create_backup(directory, backup_dir)
    
    # Rotate backups to keep only recent ones
    rotate_backups(backup_dir)
    
    print("Scheduled backup completed successfully!")

# ---- Main Script ----

# main.py
def main():
    """
    Main entry point for setting up the technical infrastructure.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Exoplanet Habitability Analysis - Technical Setup")
    parser.add_argument("--create-structure", action="store_true", help="Create project directory structure")
    parser.add_argument("--init-git", action="store_true", help="Initialize git repository")
    parser.add_argument("--backup", action="store_true", help="Create a backup of the project")
    
    args = parser.parse_args()
    
    if args.create_structure:
        create_project_structure()
    
    if args.init_git:
        setup_git_repository()
    
    if args.backup:
        scheduled_backup()
    
    if not any(vars(args).values()):
        print("No actions specified. Use --help to see available options.")

if __name__ == "__main__":
    main()
