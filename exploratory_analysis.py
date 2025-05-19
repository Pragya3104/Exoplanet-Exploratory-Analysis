"""
Exoplanet Exploratory Data Analysis Module
-----------------------------------------
This module handles the exploratory data analysis of exoplanet data,
including statistical summaries and visualizations.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm
import logging

logger = logging.getLogger(__name__)

class ExoplanetEDA:
    """Class for exploratory data analysis of exoplanet data"""
    
    def __init__(self, output_dir="results/eda"):
        """Initialize with output directory for saving results"""
        self.output_dir = output_dir
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def generate_descriptive_statistics(self, df):
        """
        Generate descriptive statistics for key variables
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with descriptive statistics
        """
        logger.info("Generating descriptive statistics")
        
        # Identify numerical columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate statistics
        stats_df = df[numeric_cols].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        
        # Add additional statistics
        stats_df.loc['skew'] = df[numeric_cols].skew()
        stats_df.loc['kurtosis'] = df[numeric_cols].kurtosis()
        stats_df.loc['missing'] = df[numeric_cols].isnull().sum()
        stats_df.loc['missing_pct'] = df[numeric_cols].isnull().mean() * 100
        
        # Save to CSV
        stats_file = os.path.join(self.output_dir, "descriptive_statistics.csv")
        stats_df.to_csv(stats_file)
        logger.info(f"Saved descriptive statistics to {stats_file}")
        
        return stats_df
    
    def analyze_distributions(self, df):
        """
        Analyze distributions of key planetary characteristics
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
        """
        logger.info("Analyzing distributions of planetary characteristics")
        
        # Define key planetary variables to analyze
        planet_vars = ['pl_rade', 'pl_bmasse', 'pl_orbper', 'pl_orbeccen', 'pl_eqt']
        planet_var_names = {
            'pl_rade': 'Planet Radius (Earth Radii)',
            'pl_bmasse': 'Planet Mass (Earth Masses)',
            'pl_orbper': 'Orbital Period (days)',
            'pl_orbeccen': 'Orbital Eccentricity',
            'pl_eqt': 'Equilibrium Temperature (K)'
        }
        
        # Create histogram for each variable
        for var in planet_vars:
            if var in df.columns:
                plt.figure(figsize=(10, 6))
                
                # Get data without NaNs
                data = df[var].dropna()
                
                # Skip if no data
                if len(data) == 0:
                    logger.warning(f"No data available for {var}, skipping histogram")
                    continue
                
                # For highly skewed data, use log scale
                if data.skew() > 2 and (data > 0).all():
                    # For orbital period and mass, which can span many orders of magnitude
                    if var in ['pl_orbper', 'pl_bmasse']:
                        bins = np.logspace(np.log10(data.min()), np.log10(data.max()), 50)
                        plt.xscale('log')
                        plt.hist(data, bins=bins, alpha=0.7)
                    else:
                        plt.hist(data, bins=50, alpha=0.7)
                else:
                    plt.hist(data, bins=50, alpha=0.7)
                
                plt.xlabel(planet_var_names.get(var, var))
                plt.ylabel('Frequency')
                plt.title(f'Distribution of {planet_var_names.get(var, var)}')
                plt.grid(alpha=0.3)
                
                # Save figure
                fig_path = os.path.join(self.output_dir, f"{var}_distribution.png")
                plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved distribution plot to {fig_path}")
    
    def analyze_correlations(self, df):
        """
        Analyze correlations between features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
            
        Returns:
        --------
        pandas.DataFrame
            Correlation matrix
        """
        logger.info("Analyzing correlations between features")
        
        # Select relevant numerical columns
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Filter to columns with less than 50% missing values
        valid_cols = [col for col in num_cols if df[col].isnull().mean() < 0.5]
        
        # Calculate correlation matrix
        corr_matrix = df[valid_cols].corr()
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix, 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            annot=True, 
            fmt=".2f", 
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Feature Correlation Matrix', fontsize=16)
        
        # Save correlation heatmap
        corr_path = os.path.join(self.output_dir, "correlation_heatmap.png")
        plt.savefig(corr_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved correlation heatmap to {corr_path}")
        
        # Save correlation matrix to CSV
        corr_csv_path = os.path.join(self.output_dir, "correlation_matrix.csv")
        corr_matrix.to_csv(corr_csv_path)
        logger.info(f"Saved correlation matrix to {corr_csv_path}")
        
        return corr_matrix
    
    def create_scatter_plots(self, df):
        """
        Create scatter plots for key relationships
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
        """
        logger.info("Creating scatter plots for key relationships")
        
        # Key relationships to visualize
        relationships = [
            # (x, y, hue, title)
            ('pl_rade', 'pl_bmasse', None, 'Mass-Radius Relationship'),
            ('pl_orbper', 'pl_eqt', None, 'Orbital Period vs Equilibrium Temperature'),
            ('st_mass', 'pl_rade', None, 'Star Mass vs Planet Radius'),
            ('st_rad', 'pl_rade', None, 'Star Radius vs Planet Radius'),
            ('insolation_flux', 'pl_eqt', None, 'Insolation Flux vs Equilibrium Temperature')
        ]
        
        for x_var, y_var, hue, title in relationships:
            # Check if variables exist in the dataframe
            if x_var not in df.columns or y_var not in df.columns:
                logger.warning(f"Variables {x_var} or {y_var} not in dataframe, skipping plot")
                continue
                
            # Create the plot
            plt.figure(figsize=(10, 8))
            
            # Handle hue variable if specified
            if hue and hue in df.columns:
                scatter = sns.scatterplot(data=df, x=x_var, y=y_var, hue=hue, alpha=0.7)
            else:
                scatter = sns.scatterplot(data=df, x=x_var, y=y_var, alpha=0.7)
            
            # For mass-radius relationship, consider log scale
            if x_var == 'pl_rade' and y_var == 'pl_bmasse':
                plt.xscale('log')
                plt.yscale('log')
                
                # Add Earth and Jupiter reference points
                plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='Earth Mass')
                plt.axhline(y=317.8, color='orange', linestyle='--', alpha=0.5, label='Jupiter Mass')
                plt.axvline(x=1, color='green', linestyle='--', alpha=0.5, label='Earth Radius')
                plt.axvline(x=11.2, color='orange', linestyle='--', alpha=0.5, label='Jupiter Radius')
                plt.legend()
            
            # For orbital period, use log scale
            if x_var == 'pl_orbper':
                plt.xscale('log')
            
            # Get axis labels
            x_label = x_var.replace('pl_', 'Planet ').replace('st_', 'Star ').replace('_', ' ').title()
            y_label = y_var.replace('pl_', 'Planet ').replace('st_', 'Star ').replace('_', ' ').title()
            
            # Special handling for common variables
            if x_var == 'pl_rade':
                x_label = 'Planet Radius (Earth Radii)'
            elif x_var == 'pl_bmasse':
                x_label = 'Planet Mass (Earth Masses)'
            elif x_var == 'pl_orbper':
                x_label = 'Orbital Period (days)'
            elif x_var == 'pl_eqt':
                x_label = 'Equilibrium Temperature (K)'
            elif x_var == 'insolation_flux':
                x_label = 'Insolation Flux (Earth Flux)'
                
            if y_var == 'pl_rade':
                y_label = 'Planet Radius (Earth Radii)'
            elif y_var == 'pl_bmasse':
                y_label = 'Planet Mass (Earth Masses)'
            elif y_var == 'pl_orbper':
                y_label = 'Orbital Period (days)'
            elif y_var == 'pl_eqt':
                y_label = 'Equilibrium Temperature (K)'
            elif y_var == 'insolation_flux':
                y_label = 'Insolation Flux (Earth Flux)'
            
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(title)
            plt.grid(alpha=0.3)
            
            # Save the plot
            plot_name = f"{x_var}_vs_{y_var}.png"
            plot_path = os.path.join(self.output_dir, plot_name)
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved scatter plot to {plot_path}")
    
    def analyze_discovery_methods(self, df):
        """
        Analyze distribution of exoplanet discovery methods
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
        """
        logger.info("Analyzing exoplanet discovery methods")
        
        # Check if discovery method column exists
        if 'discoverymethod' not in df.columns:
            logger.warning("Discovery method column not found, skipping analysis")
            return
        
        # Count discoveries by method
        method_counts = df['discoverymethod'].value_counts()
        
        # Plot bar chart of discovery methods
        plt.figure(figsize=(12, 8))
        sns.barplot(x=method_counts.index, y=method_counts.values)
        plt.xlabel('Discovery Method')
        plt.ylabel('Number of Planets')
        plt.title('Exoplanets by Discovery Method')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "discovery_methods.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved discovery methods plot to {plot_path}")
        
        # If discovery year is available, create timeline
        if 'disc_year' in df.columns:
            # Group by year and method
            timeline_data = df.groupby(['disc_year', 'discoverymethod']).size().unstack(fill_value=0)
            
            # Plot stacked bar chart
            plt.figure(figsize=(15, 8))
            timeline_data.plot(kind='bar', stacked=True, figsize=(15, 8))
            plt.xlabel('Discovery Year')
            plt.ylabel('Number of Planets')
            plt.title('Exoplanet Discoveries by Year and Method')
            plt.legend(title='Discovery Method', bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save the plot
            timeline_path = os.path.join(self.output_dir, "discovery_timeline.png")
            plt.savefig(timeline_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved discovery timeline plot to {timeline_path}")
    
    def mass_radius_diagram(self, df):
        """
        Create detailed mass-radius diagram with density contours
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
        """
        logger.info("Creating mass-radius diagram")
        
        # Check if required columns exist
        if 'pl_rade' not in df.columns or 'pl_bmasse' not in df.columns:
            logger.warning("Planet radius or mass columns not found, skipping mass-radius diagram")
            return
        
        # Filter out rows with missing values
        plot_df = df.dropna(subset=['pl_rade', 'pl_bmasse'])
        
        # Create the plot
        plt.figure(figsize=(12, 10))
        
        # Create scatter plot with density contours if enough data points
        if len(plot_df) > 50:
            # Separate gas giants, super-earths and terrestrial planets
            terrestrial = plot_df[(plot_df['pl_rade'] <= 1.6)]
            super_earth = plot_df[(plot_df['pl_rade'] > 1.6) & (plot_df['pl_rade'] <= 4)]
            gas_giant = plot_df[plot_df['pl_rade'] > 4]
            
            # Plot each category
            plt.scatter(terrestrial['pl_rade'], terrestrial['pl_bmasse'], 
                      label='Terrestrial', alpha=0.7, edgecolor='k', linewidth=0.5)
            plt.scatter(super_earth['pl_rade'], super_earth['pl_bmasse'], 
                      label='Super-Earth/Mini-Neptune', alpha=0.7, edgecolor='k', linewidth=0.5)
            plt.scatter(gas_giant['pl_rade'], gas_giant['pl_bmasse'], 
                      label='Gas Giant', alpha=0.7, edgecolor='k', linewidth=0.5)
            
            # Add density lines (constant density curves)
            # Density lines are R = (3M/4πρ)^(1/3), or M = (4πρ/3)R^3
            # For log scale: log(M) = log(4πρ/3) + 3log(R)
            r_values = np.logspace(-0.5, 1.5, 100)  # From 0.3 to 30 Earth radii
            
            # Earth density (5.51 g/cm³)
            earth_density = 5.51
            m_earth_density = (4 * np.pi * earth_density / 3) * r_values**3
            plt.plot(r_values, m_earth_density, '--', color='darkgreen', 
                   label=f'Earth density ({earth_density:.2f} g/cm³)', alpha=0.7)
            
            # Neptune density (1.64 g/cm³)
            neptune_density = 1.64
            m_neptune_density = (4 * np.pi * neptune_density / 3) * r_values**3
            plt.plot(r_values, m_neptune_density, '--', color='darkblue', 
                   label=f'Neptune density ({neptune_density:.2f} g/cm³)', alpha=0.7)
            
            # Jupiter density (1.33 g/cm³)
            jupiter_density = 1.33
            m_jupiter_density = (4 * np.pi * jupiter_density / 3) * r_values**3
            plt.plot(r_values, m_jupiter_density, '--', color='brown', 
                   label=f'Jupiter density ({jupiter_density:.2f} g/cm³)', alpha=0.7)
        else:
            # Simple scatter plot if not enough data points
            plt.scatter(plot_df['pl_rade'], plot_df['pl_bmasse'], alpha=0.7)
        
        # Add solar system planets for reference
        solar_system = {
            'Mercury': {'radius': 0.383, 'mass': 0.055},
            'Venus': {'radius': 0.949, 'mass': 0.815},
            'Earth': {'radius': 1.0, 'mass': 1.0},
            'Mars': {'radius': 0.532, 'mass': 0.107},
            'Jupiter': {'radius': 11.21, 'mass': 317.8},
            'Saturn': {'radius': 9.45, 'mass': 95.2},
            'Uranus': {'radius': 4.01, 'mass': 14.5},
            'Neptune': {'radius': 3.88, 'mass': 17.1}
        }
        
        for planet, data in solar_system.items():
            plt.scatter(data['radius'], data['mass'], marker='*', s=100, 
                      label=planet if planet in ['Earth', 'Jupiter'] else None)
            plt.annotate(planet, (data['radius'], data['mass']), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Set log scales
        plt.xscale('log')
        plt.yscale('log')
        
        # Add labels and title
        plt.xlabel('Planet Radius (Earth Radii)')
        plt.ylabel('Planet Mass (Earth Masses)')
        plt.title('Mass-Radius Diagram of Exoplanets')
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "mass_radius_diagram.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved mass-radius diagram to {plot_path}")
    
    def run_complete_eda(self, df):
        """
        Run a complete exploratory data analysis
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame containing exoplanet data
            
        Returns:
        --------
        dict
            Dictionary containing key analysis results
        """
        logger.info("Running complete exploratory data analysis")
        
        # Create results dictionary
        results = {}
        
        # Generate descriptive statistics
        results['statistics'] = self.generate_descriptive_statistics(df)
        
        # Analyze distributions
        self.analyze_distributions(df)
        
        # Analyze correlations
        results['correlations'] = self.analyze_correlations(df)
        
        # Create scatter plots
        self.create_scatter_plots(df)
        
        # Analyze discovery methods
        self.analyze_discovery_methods(df)
        
        # Create mass-radius diagram
        self.mass_radius_diagram(df)
        
        logger.info("Completed exploratory data analysis")
        
        return results


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test with some dummy data
    from data_acquisition import ExoplanetDataAcquisition
    
    data_acquisition = ExoplanetDataAcquisition()
    df = data_acquisition.prepare_data_for_analysis()
    
    eda = ExoplanetEDA()
    eda.run_complete_eda(df)
    
    print("EDA completed and results saved to the output directory.")