"""
Interactive Visualization Module for Exoplanet Exploration Project

This module creates interactive visualizations for exploring exoplanet data
using Plotly and Dash to enable dynamic filtering, comparative analysis,
and habitable zone visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExoplanetDashboard:
    """Class for creating an interactive dashboard for exoplanet data visualization."""
    
    def __init__(self, exoplanet_df: pd.DataFrame, model=None):
        """
        Initialize the dashboard with exoplanet data and optional model.
        
        Args:
            exoplanet_df: DataFrame containing exoplanet data
            model: Optional trained machine learning model for predictions
        """
        self.df = exoplanet_df.copy()
        self.model = model
        self._preprocess_data()
        self.app = self._create_app()
        logger.info("Dashboard initialized with %d exoplanets", len(self.df))
    
    def _preprocess_data(self) -> None:
        """Preprocess data for visualization."""
        # Fill missing values with NaN for better visualization handling
        self.df = self.df.replace('', np.nan)
        
        # Create discovery year column if it doesn't exist
        if 'disc_year' not in self.df.columns and 'disc_pubdate' in self.df.columns:
            try:
                self.df['disc_year'] = pd.to_datetime(self.df['disc_pubdate']).dt.year
                logger.info("Created disc_year from disc_pubdate")
            except:
                logger.warning("Could not create disc_year from disc_pubdate")
        
        # Create log versions of some columns for better visualization
        for col in ['pl_orbper', 'pl_rade', 'pl_bmasse', 'st_dist']:
            if col in self.df.columns:
                self.df[f'log_{col}'] = np.log10(pd.to_numeric(self.df[col], errors='coerce'))
                
        # Create temperature in Celsius if available in Kelvin
        if 'pl_eqt' in self.df.columns:
            self.df['pl_eqt_celsius'] = pd.to_numeric(self.df['pl_eqt'], errors='coerce') - 273.15
    
    def _get_numeric_columns(self) -> List[str]:
        """Get list of numeric columns suitable for visualization."""
        numeric_cols = self.df.select_dtypes(include=['number']).columns.tolist()
        # Filter out columns with too many NaN values
        valid_cols = [col for col in numeric_cols 
                     if self.df[col].notna().sum() > len(self.df) * 0.1]
        return valid_cols
    
    def _get_categorical_columns(self) -> List[str]:
        """Get list of categorical columns suitable for filtering."""
        object_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        # Add boolean columns converted to strings
        bool_cols = self.df.select_dtypes(include=['bool']).columns.tolist()
        
        # Filter out columns with too many unique values
        valid_cols = [col for col in object_cols + bool_cols 
                     if col in self.df.columns and 
                     self.df[col].nunique() < 100 and 
                     self.df[col].nunique() > 1]
        
        return valid_cols
    
    def _create_app(self) -> dash.Dash:
        """
        Create the Dash application with layout and callbacks.
        
        Returns:
            Configured Dash application
        """
        app = dash.Dash(__name__, external_stylesheets=[dbc.themes.COSMO])
        
        # Get columns for dropdowns
        numeric_cols = self._get_numeric_columns()
        categorical_cols = self._get_categorical_columns()
        
        # Default selection for scatter plot
        default_x = 'pl_orbper' if 'pl_orbper' in numeric_cols else numeric_cols[0]
        default_y = 'pl_rade' if 'pl_rade' in numeric_cols else numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        
        # Create app layout
        app.layout = dbc.Container([
            dbc.Row([
                dbc.Col(html.H1("Exoplanet Exploration Dashboard", className="text-center mb-4"), width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Filters", className="mb-3"),
                    html.Div([
                        html.Label("Discovery Year Range:"),
                        dcc.RangeSlider(
                            id='year-slider',
                            min=int(self.df['disc_year'].min()) if 'disc_year' in self.df.columns else 1990,
                            max=int(self.df['disc_year'].max()) if 'disc_year' in self.df.columns else 2024,
                            value=[
                                int(self.df['disc_year'].min()) if 'disc_year' in self.df.columns else 1990,
                                int(self.df['disc_year'].max()) if 'disc_year' in self.df.columns else 2024
                            ],
                            marks={i: str(i) for i in range(
                                int(self.df['disc_year'].min()) if 'disc_year' in self.df.columns else 1990,
                                int(self.df['disc_year'].max()) if 'disc_year' in self.df.columns else 2024,
                                5
                            )},
                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-4"),
                    
                    
                    
                    html.Div([
                        html.Label("Discovery Method:"),
                        dcc.Dropdown(
                            id='discovery-method-dropdown',
                            options=[
                                {'label': 'All', 'value': 'All'}
                            ] + [
                                {'label': str(val), 'value': str(val)}
                                for val in sorted(self.df['discoverymethod'].dropna().unique())
                                if 'discoverymethod' in self.df.columns
                            ],
                            value='All',
                            multi=True
                        )
                    ], className="mb-4"),
                    
                    html.Div([
                        html.Label("Additional Filter:"),
                        dcc.Dropdown(
                            id='filter-column-dropdown',
                            options=[{'label': col, 'value': col} for col in categorical_cols],
                            value=None
                        ),
                        dcc.Dropdown(
                            id='filter-value-dropdown',
                            options=[],
                            value=None,
                            multi=True
                        )
                    ], className="mb-4")
                ], width=3),
                
                dbc.Col([
                    html.H4("Exoplanet Scatter Plot", className="mb-3"),
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                html.Label("X-Axis:"),
                                dcc.Dropdown(
                                    id='x-axis-dropdown',
                                    options=[{'label': col, 'value': col} for col in numeric_cols],
                                    value=default_x
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Y-Axis:"),
                                dcc.Dropdown(
                                    id='y-axis-dropdown',
                                    options=[{'label': col, 'value': col} for col in numeric_cols],
                                    value=default_y
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Color By:"),
                                dcc.Dropdown(
                                    id='color-dropdown',
                                    options=[{'label': 'None', 'value': 'None'}] + 
                                            [{'label': col, 'value': col} for col in categorical_cols + numeric_cols],
                                    value='discoverymethod' if 'discoverymethod' in self.df.columns else 'None'
                                )
                            ], width=4)
                        ], className="mb-2"),
                        
                        dbc.Row([
                            dbc.Col([
                                html.Label("Log X-Axis:"),
                                dcc.Checklist(
                                    id='log-x-checkbox',
                                    options=[{'label': ' Enable', 'value': 'log_x'}],
                                    value=['log_x'] if default_x in ['pl_orbper', 'pl_bmasse'] else []
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Log Y-Axis:"),
                                dcc.Checklist(
                                    id='log-y-checkbox',
                                    options=[{'label': ' Enable', 'value': 'log_y'}],
                                    value=[]
                                )
                            ], width=4),
                            dbc.Col([
                                html.Label("Show Habitable Zone:"),
                                dcc.Checklist(
                                    id='habitable-zone-checkbox',
                                    options=[{'label': ' Enable', 'value': 'show_hz'}],
                                    value=[]
                                )
                            ], width=4)
                        ], className="mb-3")
                    ]),
                    
                    dcc.Graph(id='exoplanet-scatter', style={'height': '600px'}),
                    
                    html.Div(id='selection-info', className="mt-3")
                ], width=9)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Exoplanet Discoveries Timeline", className="mb-3 mt-5"),
                    dcc.Graph(id='discoveries-timeline')
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.H4("Exoplanet Characteristics Distribution", className="mb-3 mt-5"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Characteristic:"),
                            dcc.Dropdown(
                                id='distribution-column-dropdown',
                                options=[{'label': col, 'value': col} for col in numeric_cols],
                                value='pl_rade' if 'pl_rade' in numeric_cols else numeric_cols[0]
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Group By:"),
                            dcc.Dropdown(
                                id='group-by-dropdown',
                                options=[{'label': 'None', 'value': 'None'}] + 
                                        [{'label': col, 'value': col} for col in categorical_cols],
                                value='None'
                            )
                        ], width=4),
                        dbc.Col([
                            html.Label("Plot Type:"),
                            dcc.RadioItems(
                                id='distribution-plot-type',
                                options=[
                                    {'label': 'Histogram', 'value': 'histogram'},
                                    {'label': 'Box Plot', 'value': 'box'},
                                    {'label': 'Violin Plot', 'value': 'violin'}
                                ],
                                value='histogram',
                                inline=True
                            )
                        ], width=4)
                    ], className="mb-3"),
                    dcc.Graph(id='distribution-plot')
                ], width=12)
            ]),
            
            # Add section for model predictions if model is provided
            dbc.Row([
                dbc.Col([
                    html.H4("Habitable Zone Prediction", className="mb-3 mt-5"),
                    html.Div([
                        html.P("Select parameter values to predict habitability potential:"),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Stellar Temperature (K):"),
                                dcc.Input(
                                    id='stellar-temp-input',
                                    type='number',
                                    value=5778,  # Sun-like
                                    min=2000,
                                    max=40000,
                                    step=100
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Planet Radius (Earth Radii):"),
                                dcc.Input(
                                    id='planet-radius-input',
                                    type='number',
                                    value=1.0,
                                    min=0.1,
                                    max=15,
                                    step=0.1
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Orbital Period (days):"),
                                dcc.Input(
                                    id='orbital-period-input',
                                    type='number',
                                    value=365,
                                    min=0.1,
                                    max=10000,
                                    step=1
                                )
                            ], width=3),
                            dbc.Col([
                                html.Label("Stellar Radius (Solar Radii):"),
                                dcc.Input(
                                    id='stellar-radius-input',
                                    type='number',
                                    value=1.0,
                                    min=0.1,
                                    max=100,
                                    step=0.1
                                )
                            ], width=3)
                        ], className="mb-3"),
                        dbc.Button("Predict Habitability", id="predict-button", color="primary", className="mb-3"),
                        html.Div(id="prediction-result", className="mt-3")
                    ], id="prediction-container", style={'display': 'block' if self.model else 'none'})
                ], width=12)
            ]),
            
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.P(f"Data last updated: {datetime.now().strftime('%Y-%m-%d')}",
                          className="text-muted small text-center")
                ], width=12)
            ])
        ], fluid=True)
        
        # Define callbacks
        self._create_callbacks(app)
        
        return app
    
    def _create_callbacks(self, app: dash.Dash) -> None:
        """
        Create all callbacks for interactive dashboard.
        
        Args:
            app: Dash application instance
        """
        @app.callback(
            Output('filter-value-dropdown', 'options'),
            [Input('filter-column-dropdown', 'value')]
        )
        def update_filter_values(column):
            if not column:
                return []
            
            values = sorted(self.df[column].dropna().unique())
            return [{'label': str(val), 'value': str(val)} for val in values]
        
        @app.callback(
            [Output('exoplanet-scatter', 'figure'),
             Output('selection-info', 'children')],
            [Input('x-axis-dropdown', 'value'),
             Input('y-axis-dropdown', 'value'),
             Input('color-dropdown', 'value'),
             Input('log-x-checkbox', 'value'),
             Input('log-y-checkbox', 'value'),
             Input('habitable-zone-checkbox', 'value'),
             Input('year-slider', 'value'),
             Input('discovery-method-dropdown', 'value'),
             Input('filter-column-dropdown', 'value'),
             Input('filter-value-dropdown', 'value'),
             Input('exoplanet-scatter', 'clickData')]
        )
        def update_scatter_plot(x_col, y_col, color_col, log_x, log_y, show_hz, 
                             year_range, star_types, discovery_methods, 
                             filter_col, filter_values):
            # Filter data
            filtered_df = self._filter_dataframe(
                year_range, star_types, discovery_methods, filter_col, filter_values
            )
            
            # Create scatter plot
            fig = self._create_scatter_plot(
                 filtered_df, x_col, y_col, color_col, 
    '            log_x' in log_x, 'log_y' in log_y, 'show_hz' in show_hz
            )

            # Get info about selected point
            selection_info = html.Div("Click on a point to see details.")

            return fig, selection_info  # ✅ FIXED

            
        @app.callback(
            Output('discoveries-timeline', 'figure'),
            [Input('year-slider', 'value'),
             Input('discovery-method-dropdown', 'value'),
             Input('filter-column-dropdown', 'value'),
             Input('filter-value-dropdown', 'value')]
        )
        def update_timeline(year_range, discovery_methods, filter_col, filter_values):
            # Filter data
            filtered_df = self._filter_dataframe(
                year_range, None, discovery_methods, filter_col, filter_values
            )
            
            # Create timeline visualization
            return self._create_timeline_visualization(filtered_df)
        
        @app.callback(
            Output('distribution-plot', 'figure'),
            [Input('distribution-column-dropdown', 'value'),
             Input('group-by-dropdown', 'value'),
             Input('distribution-plot-type', 'value'),
             Input('year-slider', 'value'),
             Input('discovery-method-dropdown', 'value'),
             Input('filter-column-dropdown', 'value'),
             Input('filter-value-dropdown', 'value')]
        )
        def update_distribution_plot(column, group_by, plot_type, year_range, 
                                  discovery_methods, filter_col, filter_values):
            # Filter data
            # We’re not filtering by star type in this callback, so pass None
            filtered_df = self._filter_dataframe(
                 year_range,               # year_range
                 None,                     # star_types (none selected here)
                 discovery_methods,        # discovery_methods
                 filter_col,               # filter_col
                 filter_values             # filter_values
            )

            
            # Create distribution plot
            return self._create_distribution_plot(filtered_df, column, group_by, plot_type)
        
        if self.model:
            @app.callback(
                Output('prediction-result', 'children'),
                [Input('predict-button', 'n_clicks')],
                [State('stellar-temp-input', 'value'),
                 State('planet-radius-input', 'value'),
                 State('orbital-period-input', 'value'),
                 State('stellar-radius-input', 'value')]
            )
            def predict_habitability(n_clicks, stellar_temp, planet_radius, orbital_period, stellar_radius):
                if n_clicks is None:
                    return html.Div()
                
                try:
                    # Prepare input for model prediction
                    features = pd.DataFrame({
                        'st_teff': [float(stellar_temp)],
                        'pl_rade': [float(planet_radius)],
                        'pl_orbper': [float(orbital_period)],
                        'st_rad': [float(stellar_radius)]
                    })
                    
                    # Make prediction
                    prediction = self.model.predict(features)[0]
                    prob_habitable = prediction[1] if len(prediction) > 1 else prediction[0]
                    
                    # Create result card
                    card = dbc.Card(
                        dbc.CardBody([
                            html.H5("Habitability Assessment", className="card-title"),
                            html.P(f"Probability of being in habitable zone: {prob_habitable:.2%}"),
                            dbc.Progress(value=int(prob_habitable * 100), 
                                        color="success" if prob_habitable > 0.5 else "warning",
                                        className="mb-3"),
                            html.P("Factors considered in prediction:", className="mt-3"),
                            html.Ul([
                                html.Li(f"Stellar Temperature: {stellar_temp} K"),
                                html.Li(f"Planet Radius: {planet_radius} Earth radii"),
                                html.Li(f"Orbital Period: {orbital_period} days"),
                                html.Li(f"Stellar Radius: {stellar_radius} Solar radii")
                            ])
                        ]),
                        className="mt-3"
                    )
                    return card
                    
                except Exception as e:
                    return html.Div([
                        html.P(f"Error making prediction: {str(e)}", className="text-danger")
                    ])
    
    def _filter_dataframe(self, year_range, star_types, discovery_methods, filter_col, filter_values) -> pd.DataFrame:
        """
        Filter dataframe based on user selections.
        
        Args:
            year_range: Range of years to include
            star_types: List of star types to include
            discovery_methods: List of discovery methods to include
            filter_col: Additional column to filter on
            filter_values: Values to include for filter_col
            
        Returns:
            Filtered DataFrame
        """
        df = self.df.copy()
        
        # Filter by year
        if 'disc_year' in df.columns and year_range:
            df = df[(df['disc_year'] >= year_range[0]) & (df['disc_year'] <= year_range[1])]
        
        # Filter by star type
        if 'st_spectype' in df.columns and star_types and star_types != 'All' and star_types != ['All']:
            if isinstance(star_types, list):
                if 'All' not in star_types:
                    df = df[df['st_spectype'].isin(star_types)]
            else:
                if star_types != 'All':
                    df = df[df['st_spectype'] == star_types]
        
        # Filter by discovery method
        if 'discoverymethod' in df.columns and discovery_methods and discovery_methods != 'All' and discovery_methods != ['All']:
            if isinstance(discovery_methods, list):
                if 'All' not in discovery_methods:
                    df = df[df['discoverymethod'].isin(discovery_methods)]
            else:
                if discovery_methods != 'All':
                    df = df[df['discoverymethod'] == discovery_methods]
        
        # Apply additional filter if specified
        if filter_col and filter_values and filter_values != []:
            df = df[df[filter_col].isin(filter_values)]
        
        return df
    
    def _create_scatter_plot(self, df: pd.DataFrame, x_col: str, y_col: str, 
                          color_col: str, log_x: bool, log_y: bool, show_hz: bool) -> go.Figure:
        """
        Create scatter plot of exoplanet data with specified axes and color.
        
        Args:
            df: DataFrame containing filtered data
            x_col: Column to use for x-axis
            y_col: Column to use for y-axis
            color_col: Column to use for color
            log_x: Whether to use log scale for x-axis
            log_y: Whether to use log scale for y-axis
            show_hz: Whether to show habitable zone overlay
            
        Returns:
            Plotly figure object
        """
        # Handle missing data
        plot_df = df.dropna(subset=[x_col, y_col])
        
        if len(plot_df) == 0:
            # Create empty figure with message if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for selected parameters",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Determine if color column is numeric or categorical
        color_is_numeric = (color_col != 'None') and (color_col in df.select_dtypes(include=['number']).columns)
        
        # Create base scatter plot
        if color_col == 'None':
            fig = px.scatter(
                plot_df, x=x_col, y=y_col,
                hover_name='pl_name' if 'pl_name' in plot_df.columns else None,
                log_x=log_x, log_y=log_y,
                color_discrete_sequence=['rgba(65, 105, 225, 0.7)']
            )
        else:
            color_scale = 'Viridis' if color_is_numeric else None
            fig = px.scatter(
                plot_df, x=x_col, y=y_col, 
                color=color_col,
                hover_name='pl_name' if 'pl_name' in plot_df.columns else None,
                log_x=log_x, log_y=log_y,
                color_continuous_scale=color_scale if color_is_numeric else None
            )
        
        # Add habitable zone overlay if requested
        if show_hz and x_col == 'pl_orbper' and 'st_teff' in plot_df.columns:
            # Simplified habitable zone calculation based on stellar temperature
            hz_df = self._calculate_habitable_zone(plot_df)
            
            if not hz_df.empty and 'inner_hz' in hz_df.columns and 'outer_hz' in hz_df.columns:
                # Add shaded region for habitable zone
                for _, row in hz_df.iterrows():
                    fig.add_shape(
                        type="rect",
                        x0=row['inner_hz'],
                        x1=row['outer_hz'],
                        y0=0,  # Bottom of plot
                        y1=1,  # Top of plot
                        xref="x",
                        yref="paper",
                        fillcolor="rgba(0, 255, 0, 0.1)",
                        line=dict(width=0),
                        layer="below"
                    )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            title=f"{y_col} vs {x_col} for Exoplanets",
            xaxis_title=x_col + (" (log scale)" if log_x else ""),
            yaxis_title=y_col + (" (log scale)" if log_y else ""),
            height=600,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        # Improve marker appearance
        fig.update_traces(
            marker=dict(
                size=10,
                opacity=0.7,
                line=dict(width=1, color='DarkSlateGrey')
            )
        )
        
        return fig
    
    def _calculate_habitable_zone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate habitable zone boundaries based on stellar properties.
        
        Args:
            df: DataFrame with stellar properties
            
        Returns:
            DataFrame with inner and outer habitable zone boundaries
        """
        # Simple habitable zone calculation based on stellar temperature
        # This is a simplified model for visualization purposes
        
        try:
            # Group by star to avoid duplicate calculations
            unique_stars = df[['st_teff', 'st_rad', 'st_mass']].drop_duplicates().dropna()
            
            if len(unique_stars) == 0:
                return pd.DataFrame()
                
            # Calculate luminosity relative to Sun (approximation)
            unique_stars['st_lum'] = (unique_stars['st_rad'] ** 2) * ((unique_stars['st_teff'] / 5778) ** 4)
            
            # Calculate habitable zone boundaries (in AU)
            # Using simplified formulae from Kopparapu et al. 2013
            unique_stars['inner_hz_au'] = np.sqrt(unique_stars['st_lum'] / 1.1)  # Conservative inner edge
            unique_stars['outer_hz_au'] = np.sqrt(unique_stars['st_lum'] / 0.32)  # Conservative outer edge
            
            # Convert AU to orbital period (in days) using Kepler's 3rd Law
            unique_stars['inner_hz'] = 365.25 * np.sqrt(unique_stars['inner_hz_au'] ** 3 / unique_stars['st_mass'])
            unique_stars['outer_hz'] = 365.25 * np.sqrt(unique_stars['outer_hz_au'] ** 3 / unique_stars['st_mass'])
            
            return unique_stars[['inner_hz', 'outer_hz']]
            
        except Exception as e:
            logger.error(f"Error calculating habitable zone: {str(e)}")
            return pd.DataFrame()
    
    def _create_timeline_visualization(self, df: pd.DataFrame) -> go.Figure:
        """
        Create timeline visualization of exoplanet discoveries.
        
        Args:
            df: Filtered DataFrame
            
        Returns:
            Plotly figure object
        """
        if 'disc_year' not in df.columns or df.empty:
            # Create empty figure with message if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No timeline data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Count discoveries by year
        yearly_counts = df['disc_year'].value_counts().sort_index().reset_index()
        yearly_counts.columns = ['year', 'discoveries']
        
        # Calculate cumulative discoveries
        yearly_counts['cumulative'] = yearly_counts['discoveries'].cumsum()
        
        # Create subplot with two y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add bar chart for yearly discoveries
        fig.add_trace(
            go.Bar(
                x=yearly_counts['year'],
                y=yearly_counts['discoveries'],
                name="Annual Discoveries",
                marker_color='royalblue',
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Add line chart for cumulative discoveries
        fig.add_trace(
            go.Scatter(
                x=yearly_counts['year'],
                y=yearly_counts['cumulative'],
                name="Cumulative Discoveries",
                marker_color='firebrick',
                line=dict(width=3)
            ),
            secondary_y=True
        )
        
        # Add discovery method breakdown if available
        if 'discoverymethod' in df.columns:
            method_by_year = pd.crosstab(df['disc_year'], df['discoverymethod'])
            methods = method_by_year.columns
            
            for method in methods:
                if method_by_year[method].sum() > len(df) * 0.05:  # Only show major methods
                    yearly_method = method_by_year.reset_index()
                    fig.add_trace(
                        go.Scatter(
                            x=yearly_method['disc_year'],
                            y=yearly_method[method],
                            name=f"{method}",
                            mode='lines+markers',
                            marker=dict(size=8),
                            line=dict(width=2, dash='dot'),
                            visible='legendonly'  # Hidden by default
                        ),
                        secondary_y=False
                    )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            title="Exoplanet Discoveries Timeline",
            xaxis_title="Year",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=10, r=10, t=50, b=10),
            barmode='stack'
        )
        
        # Update y-axes titles
        fig.update_yaxes(title_text="Annual Discoveries", secondary_y=False)
        fig.update_yaxes(title_text="Cumulative Discoveries", secondary_y=True)
        
        return fig
    
    def _create_distribution_plot(self, df: pd.DataFrame, column: str, 
                               group_by: str, plot_type: str) -> go.Figure:
        """
        Create distribution plot for a selected column.
        
        Args:
            df: Filtered DataFrame
            column: Column to visualize distribution
            group_by: Column to group by (optional)
            plot_type: Type of plot (histogram, box, violin)
            
        Returns:
            Plotly figure object
        """
        if column not in df.columns or df.empty:
            # Create empty figure with message if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No distribution data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # Filter out NaN values
        plot_df = df.dropna(subset=[column])
        
        # Create appropriate plot based on type
        if plot_type == 'histogram':
            if group_by != 'None' and group_by in df.columns:
                fig = px.histogram(
                    plot_df, x=column, color=group_by,
                    marginal="rug", opacity=0.7,
                    barmode='overlay',
                    histnorm='probability density'
                )
            else:
                fig = px.histogram(
                    plot_df, x=column,
                    marginal="rug", opacity=0.7,
                    histnorm='probability density'
                )
                
        elif plot_type == 'box':
            if group_by != 'None' and group_by in df.columns:
                fig = px.box(
                    plot_df, y=column, x=group_by,
                    points="all", notched=True
                )
            else:
                fig = px.box(
                    plot_df, y=column,
                    points="all", notched=True
                )
                
        elif plot_type == 'violin':
            if group_by != 'None' and group_by in df.columns:
                fig = px.violin(
                    plot_df, y=column, x=group_by,
                    box=True, points="all"
                )
            else:
                fig = px.violin(
                    plot_df, y=column,
                    box=True, points="all"
                )
        
        # Update layout
        fig.update_layout(
            template='plotly_white',
            title=f"Distribution of {column}" + (f" by {group_by}" if group_by != 'None' else ""),
            height=500,
            margin=dict(l=10, r=10, t=50, b=10)
        )
        
        return fig
    
    def _create_point_info_card(self, point_data: pd.Series) -> html.Div:
        """
        Create info card for clicked point on scatter plot.
        
        Args:
            point_data: Series with data for selected point
            
        Returns:
            HTML div with formatted info
        """
        # Select relevant columns to display
        important_cols = [
            'pl_name', 'hostname', 'discoverymethod', 'disc_year',
            'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_eqt',
            'st_teff', 'st_rad', 'st_mass', 'st_dist'
        ]
        
        # Filter to columns that exist in the data
        cols_to_show = [col for col in important_cols if col in point_data.index and pd.notna(point_data[col])]
        
        if not cols_to_show:
            return html.Div("No detailed information available for this point.")
        
        # Create card with planet information
        title = point_data.get('pl_name', 'Unknown Planet')
        
        # Format each row of information
        info_rows = []
        for col in cols_to_show:
            if col == 'pl_name':  # Skip name since it's in the title
                continue
                
            # Format value based on type
            value = point_data[col]
            if isinstance(value, (int, float)):
                formatted_value = f"{value:.4g}"
            else:
                formatted_value = str(value)
                
            # Add row with label and value
            info_rows.append(
                html.Div([
                    html.Span(f"{col}: ", className="font-weight-bold"),
                    html.Span(formatted_value)
                ], className="d-block")
            )
        
        card = dbc.Card(
            dbc.CardBody([
                html.H5(title, className="card-title"),
                html.Div(info_rows)
            ])
        )
        
        return card
    
    def run_server(self, debug: bool = True, port: int = 8050, host: str = '0.0.0.0') -> None:
        """
        Run the Dash server.
        
        Args:
            debug: Whether to run in debug mode
            port: Port to run server on
            host: Host to run server on
        """
        self.app.run(debug=debug, port=port, host=host)


def create_dashboard(exoplanet_df: pd.DataFrame, model=None, run: bool = False) -> ExoplanetDashboard:
    """
    Create and optionally run the exoplanet dashboard.
    
    Args:
        exoplanet_df: DataFrame with exoplanet data
        model: Optional trained ML model for predictions
        run: Whether to run the server immediately
        
    Returns:
        Dashboard object
    """
    dashboard = ExoplanetDashboard(exoplanet_df, model)
    
    if run:
        dashboard.run_server()
        
    return dashboard


import sys
import logging
from data_acquisition import ExoplanetDataAcquisition
from model_development import ExoplanetModel

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# --- Dashboard Launch Entry Point ---
if __name__ == "__main__":
    try:
        # Step 1: Prepare data
        data_loader = ExoplanetDataAcquisition(cache_dir="data")
        exoplanet_df = data_loader.prepare_data_for_analysis(use_cache=True)

        # Step 2: Load trained model if exists
        try:

            from model_development import ExoplanetModel, plot_roc_curve, plot_precision_recall_curve
            model = ExoplanetModel(model_type='random_forest')
            import pandas as pd
            df = pd.read_csv('data/prepared_data.csv')
            # Ensure 'habitable' column exists; create it if missing (example logic, adjust as needed)
            if 'habitable' not in df.columns:
                # Example: mark as habitable if planet is in habitable zone (customize as needed)
                df['habitable'] = ((df['pl_eqt'] >= 273.15) & (df['pl_eqt'] <= 373.15)).astype(int) if 'pl_eqt' in df.columns else 0
            X_train, X_test, y_train, y_test = model.prepare_data(df, target='habitable')
            results = model.train(X_train, y_train, X_test, y_test)
            plot_roc_curve(results, title="Model ROC Curve")

        except Exception as e:
            logger.warning(f"No pre-trained model found or failed to load: {e}")
            model = None

        # Step 3: Launch dashboard
        from interactive_visualization import create_dashboard
        dashboard = create_dashboard(exoplanet_df, model, run=True)

    except ImportError as e:
        logger.error(f"Could not import required modules: {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {str(e)}")
        sys.exit(1)
