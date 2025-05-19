"""
Habitable Zone Analysis Module

This module provides functions to analyze exoplanet habitability based on various criteria.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import logging

# Configure logger
logger = logging.getLogger(__name__)

def calculate_habitable_zone(stellar_temp: float, stellar_luminosity: float) -> Tuple[float, float]:
    """
    Calculate the inner and outer boundaries of the habitable zone based on stellar properties.
    
    Uses the conservative habitable zone model from Kopparapu et al. (2013).
    
    Args:
        stellar_temp: Effective temperature of the star in Kelvin
        stellar_luminosity: Luminosity of the star in solar units (L/L_sun)
    
    Returns:
        Tuple containing (inner_boundary, outer_boundary) in AU
    """
    logger.info(f"Calculating habitable zone for star with T={stellar_temp}K, L={stellar_luminosity}L_sun")
    
    # Conservative HZ limits coefficients (Kopparapu et al. 2013)
    # Inner edge (runaway greenhouse)
    inner_coeff = [1.1066, 1.3599e-4, 3.1631e-9, -3.1913e-12]
    # Outer edge (maximum greenhouse)
    outer_coeff = [0.3240, 5.3221e-5, 1.4288e-9, -1.1192e-12]
    
    # Normalized stellar temperature
    t_star = stellar_temp - 5780
    
    # Calculate effective solar flux at HZ boundaries
    inner_flux = inner_coeff[0] + inner_coeff[1]*t_star + inner_coeff[2]*t_star**2 + inner_coeff[3]*t_star**3
    outer_flux = outer_coeff[0] + outer_coeff[1]*t_star + outer_coeff[2]*t_star**2 + outer_coeff[3]*t_star**3
    
    # Convert to distance in AU
    inner_edge = np.sqrt(stellar_luminosity / inner_flux)
    outer_edge = np.sqrt(stellar_luminosity / outer_flux)
    
    logger.debug(f"Calculated HZ: inner={inner_edge:.3f} AU, outer={outer_edge:.3f} AU")
    return inner_edge, outer_edge

def estimate_equilibrium_temperature(
    semi_major_axis: float, 
    stellar_temp: float, 
    stellar_radius: float, 
    albedo: float = 0.3
) -> float:
    """
    Estimate the equilibrium temperature of a planet.
    
    Args:
        semi_major_axis: Semi-major axis in AU
        stellar_temp: Effective temperature of the star in Kelvin
        stellar_radius: Radius of the star in solar radii
        albedo: Bond albedo of the planet (default: 0.3, Earth-like)
    
    Returns:
        Equilibrium temperature in Kelvin
    """
    # Convert AU to meters
    a_meters = semi_major_axis * 1.496e11
    
    # Convert solar radii to meters
    r_star_meters = stellar_radius * 6.957e8
    
    # Stefan-Boltzmann constant
    sigma = 5.670374419e-8  # W/(m^2 K^4)
    
    # Calculate equilibrium temperature
    t_eq = stellar_temp * np.sqrt(r_star_meters / (2 * a_meters)) * (1 - albedo)**(1/4)
    
    return t_eq

def assign_habitability_labels(
    df: pd.DataFrame, 
    hz_params: Dict[str, Union[str, float]] = None
) -> pd.DataFrame:
    """
    Assign habitability labels to planets based on multiple criteria.
    
    Args:
        df: DataFrame containing exoplanet data
        hz_params: Dictionary of habitability parameters to use
            Keys can include:
            - 'temp_min': Minimum equilibrium temperature in K
            - 'temp_max': Maximum equilibrium temperature in K
            - 'mass_min': Minimum planet mass in Earth masses
            - 'mass_max': Maximum planet mass in Earth masses
            - 'radius_min': Minimum planet radius in Earth radii
            - 'radius_max': Maximum planet radius in Earth radii
    
    Returns:
        DataFrame with added habitability columns
    """
    # Default habitability parameters (Earth-like conditions)
    default_params = {
        'temp_min': 273,    # K (freezing point of water)
        'temp_max': 373,    # K (boiling point of water)
        'mass_min': 0.1,    # Earth masses (lower bound for retaining atmosphere)
        'mass_max': 10,     # Earth masses (upper bound before becoming gas-rich)
        'radius_min': 0.5,  # Earth radii
        'radius_max': 2.5,  # Earth radii (upper limit for rocky planets)
    }
    
    # Use default parameters if none provided
    if hz_params is None:
        hz_params = default_params
    else:
        # Fill in any missing parameters with defaults
        for key, value in default_params.items():
            if key not in hz_params:
                hz_params[key] = value
    
    logger.info(f"Assigning habitability labels using parameters: {hz_params}")
    
    # Create a copy of the dataframe to avoid modifying the original
    result_df = df.copy()
    
    # Check if planet is in the habitable zone based on semi-major axis
    if 'stellar_temp' in df.columns and 'stellar_luminosity' in df.columns and 'semi_major_axis' in df.columns:
        result_df['hz_inner'] = np.nan
        result_df['hz_outer'] = np.nan
        result_df['in_habitable_zone'] = False
        
        for idx, row in result_df.iterrows():
            try:
                inner, outer = calculate_habitable_zone(row['stellar_temp'], row['stellar_luminosity'])
                result_df.at[idx, 'hz_inner'] = inner
                result_df.at[idx, 'hz_outer'] = outer
                result_df.at[idx, 'in_habitable_zone'] = (row['semi_major_axis'] >= inner) and (row['semi_major_axis'] <= outer)
            except Exception as e:
                logger.warning(f"Could not calculate HZ for row {idx}: {e}")
    
    # Check temperature criteria (if equilibrium temperature exists)
    if 'equilibrium_temp' in df.columns:
        result_df['temp_habitable'] = (
            (result_df['equilibrium_temp'] >= hz_params['temp_min']) & 
            (result_df['equilibrium_temp'] <= hz_params['temp_max'])
        )
    elif 'stellar_temp' in df.columns and 'stellar_radius' in df.columns and 'semi_major_axis' in df.columns:
        # Calculate equilibrium temperature if not provided
        result_df['equilibrium_temp'] = result_df.apply(
            lambda x: estimate_equilibrium_temperature(
                x['semi_major_axis'], 
                x['stellar_temp'], 
                x['stellar_radius']
            ) if not pd.isna(x['semi_major_axis']) and not pd.isna(x['stellar_temp']) and not pd.isna(x['stellar_radius']) else np.nan,
            axis=1
        )
        result_df['temp_habitable'] = (
            (result_df['equilibrium_temp'] >= hz_params['temp_min']) & 
            (result_df['equilibrium_temp'] <= hz_params['temp_max'])
        )
    
    # Check mass criteria
    if 'planet_mass' in df.columns:
        result_df['mass_habitable'] = (
            (result_df['planet_mass'] >= hz_params['mass_min']) & 
            (result_df['planet_mass'] <= hz_params['mass_max'])
        )
    
    # Check radius criteria
    if 'planet_radius' in df.columns:
        result_df['radius_habitable'] = (
            (result_df['planet_radius'] >= hz_params['radius_min']) & 
            (result_df['planet_radius'] <= hz_params['radius_max'])
        )
    
    # Combine criteria for overall habitability score
    habitability_columns = [col for col in ['in_habitable_zone', 'temp_habitable', 'mass_habitable', 'radius_habitable'] 
                           if col in result_df.columns]
    
    if habitability_columns:
        # Count how many habitability criteria are met
        result_df['habitability_score'] = result_df[habitability_columns].sum(axis=1)
        # Mark as potentially habitable if all available criteria are met
        result_df['potentially_habitable'] = result_df['habitability_score'] == len(habitability_columns)
        
        logger.info(f"Found {result_df['potentially_habitable'].sum()} potentially habitable planets")
    else:
        logger.warning("No habitability criteria could be applied - insufficient data")
    
    return result_df

def plot_habitable_zone(df: pd.DataFrame, save_path: Optional[str] = None) -> None:
    """
    Create a visualization of planets in relation to their star's habitable zone.
    
    Args:
        df: DataFrame with habitability information
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if not all(col in df.columns for col in ['hz_inner', 'hz_outer', 'in_habitable_zone', 'semi_major_axis']):
        logger.error("Required columns for habitable zone plotting not found in dataframe")
        return
        
    plt.figure(figsize=(12, 8))
    
    # Plot planets
    plt.scatter(
        df['stellar_temp'], 
        df['semi_major_axis'], 
        c=df['in_habitable_zone'].map({True: 'green', False: 'gray'}), 
        alpha=0.7, 
        s=df['planet_radius']*20 if 'planet_radius' in df.columns else 30,
        label='Planets'
    )
    
    # Add habitable zone boundaries if available
    unique_stars = df[['stellar_temp', 'stellar_luminosity']].drop_duplicates().dropna()
    
    for _, star in unique_stars.iterrows():
        inner, outer = calculate_habitable_zone(star['stellar_temp'], star['stellar_luminosity'])
        plt.plot([star['stellar_temp'], star['stellar_temp']], [inner, outer], 'r-', alpha=0.3)
    
    # Plot aesthetics
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Stellar Temperature (K)')
    plt.ylabel('Semi-major Axis (AU)')
    plt.title('Exoplanets and Habitable Zones')
    plt.grid(True, alpha=0.3)
    plt.colorbar(plt.cm.ScalarMappable(cmap='RdYlGn'), 
                 label='In Habitable Zone')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved habitable zone plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def analyze_habitable_planets(df: pd.DataFrame) -> Dict:
    """
    Analyze the distribution and characteristics of potentially habitable planets.
    
    Args:
        df: DataFrame with habitability information
    
    Returns:
        Dictionary with habitability analysis results
    """
    if 'potentially_habitable' not in df.columns:
        logger.error("Habitability labels not found in dataframe")
        return {}
    
    habitable_planets = df[df['potentially_habitable'] == True]
    total_planets = len(df)
    habitable_count = len(habitable_planets)
    
    results = {
        'total_planets': total_planets,
        'habitable_count': habitable_count,
        'habitable_percentage': (habitable_count / total_planets) * 100 if total_planets > 0 else 0
    }
    
    # Analysis by stellar type if available
    if 'stellar_type' in df.columns:
        results['habitable_by_stellar_type'] = habitable_planets['stellar_type'].value_counts().to_dict()
        results['habitable_percentage_by_stellar_type'] = {}
        
        for stellar_type in df['stellar_type'].unique():
            type_total = len(df[df['stellar_type'] == stellar_type])
            type_habitable = len(habitable_planets[habitable_planets['stellar_type'] == stellar_type])
            results['habitable_percentage_by_stellar_type'][stellar_type] = (type_habitable / type_total) * 100 if type_total > 0 else 0
    
    # Analysis by discovery method if available
    if 'discovery_method' in df.columns:
        results['habitable_by_discovery_method'] = habitable_planets['discovery_method'].value_counts().to_dict()
    
    # Analysis of habitable planet properties
    for column in ['planet_radius', 'planet_mass', 'equilibrium_temp', 'semi_major_axis']:
        if column in df.columns:
            results[f'habitable_{column}_mean'] = habitable_planets[column].mean()
            results[f'habitable_{column}_median'] = habitable_planets[column].median()
            results[f'habitable_{column}_std'] = habitable_planets[column].std()
    
    logger.info(f"Completed habitability analysis: {habitable_count}/{total_planets} planets potentially habitable")
    return results
