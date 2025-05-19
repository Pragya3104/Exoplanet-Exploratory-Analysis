"""
Exoplanet Data Acquisition Module
---------------------------------
This module handles fetching, cleaning, and preparing exoplanet data
from the NASA Exoplanet Archive and other sources.
"""

import os
import pandas as pd
import numpy as np
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class ExoplanetDataAcquisition:
    """Class for acquiring and processing exoplanet data"""
    
    def __init__(self, cache_dir="data"):
        """Initialize with cache directory for data storage"""
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
    
    def fetch_confirmed_planets(self, use_cache=True):
        """
        Fetch confirmed exoplanets from NASA Exoplanet Archive
        
        Parameters:
        -----------
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing exoplanet data
        """
        cache_file = os.path.join(self.cache_dir, "confirmed_planets.csv")
        
        # Use cached data if available and requested
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached exoplanet data from {cache_file}")
            return pd.read_csv(cache_file)
        
        # Otherwise fetch from NASA API
        logger.info("Fetching confirmed planets data from NASA Exoplanet Archive")
        try:
            # Selected columns:
            # - Planet properties: name, orbital period, radius, mass, eccentricity, equilibrium temp
            # - Star properties: effective temperature, radius, mass
            # - Discovery information: discovery method, discovery year
            confirmed_planets = NasaExoplanetArchive.query_criteria(
                table="ps", 
                select="pl_name,hostname,pl_orbper,pl_rade,pl_bmasse,pl_orbeccen,pl_eqt,"
                       "st_teff,st_rad,st_mass,discoverymethod,disc_year,"
                       "pl_orbsmax,rowupdate"
            )
            
            # Convert to pandas DataFrame
            planets_df = confirmed_planets.to_pandas()
            
            # Cache the data
            planets_df.to_csv(cache_file, index=False)
            logger.info(f"Cached exoplanet data to {cache_file}")
            
            return planets_df
            
        except Exception as e:
            logger.error(f"Error fetching exoplanet data: {str(e)}")
            raise
    
    def fetch_stellar_data(self, use_cache=True):
        """
        Fetch stellar data for exoplanet host stars
        
        Parameters:
        -----------
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing stellar data
        """
        cache_file = os.path.join(self.cache_dir, "stellar_data.csv")
        
        if use_cache and os.path.exists(cache_file):
            logger.info(f"Loading cached stellar data from {cache_file}")
            return pd.read_csv(cache_file)
        
        logger.info("Fetching stellar data from NASA Exoplanet Archive")
        try:
            stellar_data = NasaExoplanetArchive.query_criteria(
                table="stars", 
                select="hip_name,st_teff,st_rad,st_mass,st_age,st_met,st_lum,st_logg,"
                       "st_rotp,st_bmvj,st_spstr,ra,dec,sy_dist"
            )
            
            # Convert to pandas DataFrame
            stellar_df = stellar_data.to_pandas()
            
            # Cache the data
            stellar_df.to_csv(cache_file, index=False)
            logger.info(f"Cached stellar data to {cache_file}")
            
            return stellar_df
            
        except Exception as e:
            logger.error(f"Error fetching stellar data: {str(e)}")
            raise
    
    def clean_data(self, df):
        """
        Clean exoplanet data by handling missing values and outliers
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
            
        Returns:
        --------
        pandas.DataFrame
            Cleaned DataFrame
        """
        logger.info("Cleaning exoplanet data")
        
        # Make a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Convert column names to lowercase for consistency
        df_clean.columns = [col.lower() for col in df_clean.columns]
        
        # Remove rows with no planet radius or mass
        initial_count = len(df_clean)
        df_clean = df_clean.dropna(subset=['pl_rade', 'pl_bmasse'], how='all')
        logger.info(f"Removed {initial_count - len(df_clean)} rows with no radius or mass data")
        
        # Handle outliers (e.g., remove planets with extremely large radii)
        radius_outliers = df_clean[df_clean['pl_rade'] > 30].shape[0]
        df_clean = df_clean[df_clean['pl_rade'] <= 30]
        logger.info(f"Removed {radius_outliers} radius outliers")
        
        # Fill missing equilibrium temperatures with median values
        if 'pl_eqt' in df_clean.columns:
            median_temp = df_clean['pl_eqt'].median()
            df_clean['pl_eqt'].fillna(median_temp, inplace=True)
        
        return df_clean
    
    def create_derived_features(self, df):
        """
        Create derived features from existing exoplanet data
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with additional derived features
        """
        logger.info("Creating derived features")
        
        # Make a copy to avoid modifying the original
        df_features = df.copy()
        
        # Calculate planet density (if radius and mass are available)
        if all(col in df_features.columns for col in ['pl_rade', 'pl_bmasse']):
            # Earth radius in km
            earth_radius_km = 6371.0
            # Earth mass in kg
            earth_mass_kg = 5.97e24
            
            # Convert Earth radii to volume in km^3 (V = 4/3 * pi * r^3)
            df_features['pl_vol_earth'] = (4/3) * np.pi * (df_features['pl_rade'] * earth_radius_km) ** 3
            
            # Convert Earth masses to kg
            df_features['pl_mass_kg'] = df_features['pl_bmasse'] * earth_mass_kg
            
            # Calculate density in kg/km^3 (later can convert to g/cm^3 if needed)
            df_features['pl_density'] = df_features['pl_mass_kg'] / df_features['pl_vol_earth']
            
            logger.info("Added planet density feature")
        
        # Calculate insolation flux (if star temperature, star radius, and orbital distance available)
        if all(col in df_features.columns for col in ['st_teff', 'st_rad', 'pl_orbsmax']):
            # Stefan-Boltzmann constant
            sigma = 5.670374419e-8  # W m^-2 K^-4
            
            # Calculate insolation flux relative to Earth
            # F = (R_star/D)^2 * (T_star/T_sun)^4
            # where T_sun = 5772K
            T_sun = 5772.0
            
            # Calculate flux
            df_features['insolation_flux'] = (df_features['st_rad'] / df_features['pl_orbsmax'])**2 * (df_features['st_teff'] / T_sun)**4
            
            logger.info("Added insolation flux feature")
        
        # Calculate orbital period in Earth days if not already present
        if 'pl_orbper' not in df_features.columns and all(col in df_features.columns for col in ['pl_orbsmax', 'st_mass']):
            # Kepler's Third Law: P^2 = (4*pi^2 / G*M) * a^3
            # For simplification, we calculate P^2 = a^3 / M (with appropriate units)
            # a in AU, M in Solar masses, P in years
            df_features['pl_orbper_calculated'] = np.sqrt((df_features['pl_orbsmax']**3) / df_features['st_mass'])
            # Convert from years to days
            df_features['pl_orbper_calculated'] = df_features['pl_orbper_calculated'] * 365.25
            
            logger.info("Added calculated orbital period feature")
        
        return df_features
    
    def prepare_data_for_analysis(self, use_cache=True):
        """
        Main method to prepare all data for analysis
        
        Parameters:
        -----------
        use_cache : bool
            Whether to use cached data if available
            
        Returns:
        --------
        pandas.DataFrame
            Fully prepared DataFrame for analysis
        """
        logger.info("Preparing data for analysis")
        
        # Check if prepared data is already cached
        prepared_cache = os.path.join(self.cache_dir, "prepared_data.csv")
        if use_cache and os.path.exists(prepared_cache):
            logger.info(f"Loading prepared data from cache: {prepared_cache}")
            return pd.read_csv(prepared_cache)
        
        # Fetch raw data
        planets_df = self.fetch_confirmed_planets(use_cache)
        
        # Clean the data
        planets_df = self.clean_data(planets_df)
        
        # Create derived features
        prepared_df = self.create_derived_features(planets_df)
        
        # Cache the prepared data
        prepared_df.to_csv(prepared_cache, index=False)
        logger.info(f"Cached prepared data to {prepared_cache}")
        
        return prepared_df
    
    def split_train_test(self, df, target_column, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to split
        target_column : str
            Name of the target column
        test_size : float
            Proportion of data to use for testing
        random_state : int
            Random seed for reproducibility
            
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        logger.info(f"Splitting data into train/test sets with test_size={test_size}")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Train set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
        
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the module
    data_acquisition = ExoplanetDataAcquisition(cache_dir="data")
    prepared_data = data_acquisition.prepare_data_for_analysis()
    print(f"Prepared data shape: {prepared_data.shape}")
    print(f"Columns: {prepared_data.columns.tolist()}")
