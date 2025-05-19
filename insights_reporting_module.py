"""
Exoplanet Analysis Project - Insights and Reporting Module
Modified to load both descriptive_statistics.csv and correlation_matrix.csv
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
from fpdf import FPDF

class InsightsReporting:
    def __init__(self, eda_dir="results/eda", output_dir="reports"):
        """
        Initialize the Insights and Reporting module

        Parameters:
        -----------
        eda_dir : str
            Directory containing the EDA output CSV files
        output_dir : str
            Directory where reports will be saved
        """
        self.eda_dir = eda_dir
        self.output_dir = output_dir
        self.descriptive_stats = None
        self.correlation_matrix = None
        self.insights = {}

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Load data
        self.load_data()

    def load_data(self):
        """Load the EDA output CSVs: descriptive statistics and correlation matrix"""
        try:
            desc_path = os.path.join(self.eda_dir, "descriptive_statistics.csv")
            corr_path = os.path.join(self.eda_dir, "correlation_matrix.csv")

            self.descriptive_stats = pd.read_csv(desc_path, index_col=0)
            self.correlation_matrix = pd.read_csv(corr_path, index_col=0)

            print("Descriptive statistics and correlation matrix loaded successfully.")

        except Exception as e:
            print(f"Error loading data: {e}")

    def analyze_statistics_summary(self):
        """Extract summary statistics for reporting"""
        if self.descriptive_stats is None:
            print("Descriptive statistics not loaded.")
            return

        summary = {
            "variables": self.descriptive_stats.columns.tolist(),
            "metrics": self.descriptive_stats.index.tolist(),
            "missing_values": self.descriptive_stats.loc['missing'].to_dict() if 'missing' in self.descriptive_stats.index else {}
        }
        self.insights['summary_statistics'] = summary
        return summary

    def analyze_correlation_summary(self):
        """Extract strongest correlations for reporting"""
        if self.correlation_matrix is None:
            print("Correlation matrix not loaded.")
            return

        corr_df = self.correlation_matrix.copy()
        np.fill_diagonal(corr_df.values, np.nan)  # Ignore self-correlation
        corr_series = corr_df.unstack().dropna().sort_values(key=lambda x: abs(x), ascending=False)
        top_5 = corr_series.head(5).to_dict()

        self.insights['top_correlations'] = top_5
        return top_5

    def generate_summary_report(self):
        """Generate a simple summary report PDF"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"summary_report_{timestamp}.pdf"
        report_path = os.path.join(self.output_dir, report_filename)

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16, style="B")
        pdf.cell(200, 10, "Exoplanet EDA Summary Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(200, 10, "Descriptive Statistics Summary", ln=True)
        pdf.set_font("Arial", size=10)
        if 'summary_statistics' in self.insights:
            for var, missing in self.insights['summary_statistics']['missing_values'].items():
                pdf.cell(200, 5, f"- {var}: {missing} missing values", ln=True)

        pdf.ln(10)
        pdf.set_font("Arial", size=12, style="B")
        pdf.cell(200, 10, "Top Correlations", ln=True)
        pdf.set_font("Arial", size=10)
        if 'top_correlations' in self.insights:
            for pair, value in self.insights['top_correlations'].items():
                var1, var2 = pair
                pdf.cell(200, 5, f"- {var1} vs {var2}: r = {value:.2f}", ln=True)

        pdf.output(report_path)
        print(f"Summary report generated: {report_path}")
        return report_path

    def run_all_analyses(self):
        """Run all analyses and generate the report"""
        print("Running all analyses...")
        self.analyze_statistics_summary()
        self.analyze_correlation_summary()
        print("Generating report...")
        return self.generate_summary_report()


if __name__ == "__main__":
    eda_output_dir = r"C:\\Users\\Yatharth Vashisht\\Desktop\\exoplanet exploratory data anaylysis\\results\\eda"
    reporter = InsightsReporting(eda_dir=eda_output_dir)
    reporter.run_all_analyses()
