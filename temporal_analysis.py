"""
Temporal Analysis Module for Exoplanet Exploration Project

This module performs temporal analysis on exoplanet discoveries over time,
including trends in discovery methods, characteristics of discovered planets,
and statistical analysis of discovery patterns.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """Class for analyzing exoplanet discovery trends over time."""
    
    def __init__(self, exoplanet_df: pd.DataFrame):
        """
        Initialize the temporal analyzer with exoplanet dataset.
        
        Args:
            exoplanet_df: DataFrame containing exoplanet data with discovery dates
        """
        self.df = exoplanet_df
        self._validate_data()
        logger.info("Temporal analyzer initialized with %d exoplanets", len(self.df))
    
    def _validate_data(self) -> None:
        """Validate input data contains required columns."""
        required_cols = ['disc_year', 'pl_name']
        recommended_cols = ['discoverymethod', 'pl_orbper', 'pl_rade', 'pl_bmasse', 'st_spectype']
        
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        missing_recommended = [col for col in recommended_cols if col not in self.df.columns]
        if missing_recommended:
            logger.warning(f"Missing recommended columns: {missing_recommended}")
    
    def discovery_rate_by_year(self) -> pd.DataFrame:
        """
        Calculate the number of exoplanets discovered per year.
        
        Returns:
            DataFrame with years and discovery counts
        """
        yearly_counts = self.df['disc_year'].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['year', 'discoveries']
        return yearly_counts
    
    def cumulative_discoveries(self) -> pd.DataFrame:
        """
        Calculate cumulative discoveries over time.
        
        Returns:
            DataFrame with years and cumulative discovery counts
        """
        yearly_counts = self.discovery_rate_by_year()
        yearly_counts['cumulative'] = yearly_counts['discoveries'].cumsum()
        return yearly_counts
    
    def discovery_methods_by_year(self) -> pd.DataFrame:
        """
        Analyze discovery methods used each year.
        
        Returns:
            DataFrame with pivot of years vs discovery methods
        """
        if 'discoverymethod' not in self.df.columns:
            logger.warning("Discovery method column not available")
            return pd.DataFrame()
            
        method_by_year = pd.crosstab(
            self.df['disc_year'], 
            self.df['discoverymethod']
        ).reset_index()
        
        return method_by_year
    
    def planet_characteristic_evolution(self, characteristic: str) -> pd.DataFrame:
        """
        Track the evolution of planet characteristics over time.
        
        Args:
            characteristic: Column name for the characteristic to analyze
            
        Returns:
            DataFrame with yearly statistics for the characteristic
        """
        if characteristic not in self.df.columns:
            logger.warning(f"Characteristic {characteristic} not in dataset")
            return pd.DataFrame()
            
        # Group by year and calculate statistics
        yearly_stats = self.df.groupby('disc_year')[characteristic].agg([
            'mean', 'median', 'std', 'count'
        ]).reset_index()
        
        return yearly_stats
    
    def detection_method_evolution(self) -> pd.DataFrame:
        """
        Analyze how detection methods have evolved over time.
        
        Returns:
            DataFrame with proportions of each method by year
        """
        if 'discoverymethod' not in self.df.columns:
            logger.warning("Discovery method column not available")
            return pd.DataFrame()
            
        # Get counts by year and method
        method_counts = self.discovery_methods_by_year()
        
        # Calculate proportions
        method_props = method_counts.copy()
        methods = method_props.columns.drop('disc_year')
        
        for method in methods:
            total_by_year = method_props[methods].sum(axis=1)
            method_props[f"{method}_proportion"] = method_props[method] / total_by_year
            
        return method_props
    
    def visualize_discovery_timeline(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of exoplanet discoveries over time.
        
        Args:
            save_path: Optional path to save the figure
        """
        cumulative = self.cumulative_discoveries()
        yearly = self.discovery_rate_by_year()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot yearly discoveries
        ax1.bar(yearly['year'], yearly['discoveries'], color='royalblue')
        ax1.set_title('Yearly Exoplanet Discoveries', fontsize=14)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Discoveries')
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot cumulative discoveries
        ax2.plot(cumulative['year'], cumulative['cumulative'], 
                 marker='o', linestyle='-', color='darkred')
        ax2.set_title('Cumulative Exoplanet Discoveries', fontsize=14)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Number of Discoveries')
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Discovery timeline visualization saved to {save_path}")
        
        plt.show()
    
    def visualize_method_evolution(self, save_path: Optional[str] = None) -> None:
        """
        Visualize the evolution of discovery methods over time.
        
        Args:
            save_path: Optional path to save the figure
        """
        if 'discoverymethod' not in self.df.columns:
            logger.warning("Discovery method column not available")
            return
            
        # Get method data by year
        method_data = self.discovery_methods_by_year()
        
        # Create pivot for plotting
        years = method_data['disc_year']
        methods = method_data.columns.drop('disc_year')
        
        plt.figure(figsize=(14, 8))
        
        # Create stacked area chart
        plt.stackplot(years, 
                     [method_data[method] for method in methods],
                     labels=methods, alpha=0.8)
        
        plt.title('Evolution of Exoplanet Discovery Methods', fontsize=16)
        plt.xlabel('Year', fontsize=12)
        plt.ylabel('Number of Discoveries', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper left', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Method evolution visualization saved to {save_path}")
        
        plt.show()
    
    def analyze_decade_trends(self) -> Dict:
        """
        Analyze trends by decade instead of individual years.
        
        Returns:
            Dictionary of DataFrames with decade-level analysis
        """
        # Create decade column
        self.df['decade'] = (self.df['disc_year'] // 10) * 10
        
        results = {}
        
        # Count by decade
        results['counts'] = self.df['decade'].value_counts().sort_index()
        
        # Method distribution by decade
        if 'discoverymethod' in self.df.columns:
            results['methods'] = pd.crosstab(
                self.df['decade'],
                self.df['discoverymethod']
            )
        
        # Planet characteristics by decade
        characteristics = ['pl_orbper', 'pl_rade', 'pl_bmasse']
        available_chars = [c for c in characteristics if c in self.df.columns]
        
        for char in available_chars:
            results[f'{char}_by_decade'] = self.df.groupby('decade')[char].agg([
                'mean', 'median', 'std', 'min', 'max', 'count'
            ])
            
        return results

def run_temporal_analysis(exoplanet_df: pd.DataFrame, output_dir: str) -> Dict:
    """
    Run temporal analysis on exoplanet data and save visualizations.
    
    Args:
        exoplanet_df: DataFrame with exoplanet data
        output_dir: Directory to save output files
    
    Returns:
        Dictionary with analysis results
    """
    import os
    from pathlib import Path
    
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = TemporalAnalyzer(exoplanet_df)
    
    # Run analyses
    results = {
        'yearly_discoveries': analyzer.discovery_rate_by_year(),
        'cumulative_discoveries': analyzer.cumulative_discoveries(),
        'method_evolution': analyzer.detection_method_evolution()
    }
    
    # Run characteristic evolution for common properties
    for char in ['pl_orbper', 'pl_rade', 'pl_bmasse']:
        if char in exoplanet_df.columns:
            results[f'{char}_evolution'] = analyzer.planet_characteristic_evolution(char)
    
    # Generate visualizations
    analyzer.visualize_discovery_timeline(
        save_path=os.path.join(output_dir, 'discovery_timeline.png')
    )
    
    analyzer.visualize_method_evolution(
        save_path=os.path.join(output_dir, 'method_evolution.png')
    )
    
    # Add decade analysis
    results['decade_analysis'] = analyzer.analyze_decade_trends()
    
    logger.info("Temporal analysis completed successfully")
    return results

if __name__ == "__main__":
    # Sample execution code
    import sys
    
    try:
        # Assuming data is loaded from another module
        from data_acquisition import load_exoplanet_data
        
        # Load data
        exoplanet_data = load_exoplanet_data()
        
        # Run analysis
        results = run_temporal_analysis(exoplanet_data, "./output/temporal")
        
        # Output some sample results
        print(f"Total years analyzed: {len(results['yearly_discoveries'])}")
        print(f"Peak discovery year: {results['yearly_discoveries'].sort_values('discoveries', ascending=False).iloc[0]['year']}")
        
    except ImportError:
        logger.error("Could not import required modules")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in temporal analysis: {str(e)}")
        sys.exit(1)
