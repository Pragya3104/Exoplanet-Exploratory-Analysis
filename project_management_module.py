# Project Management Module for Exoplanet Habitability Analysis

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import json
import os

class ProjectManager:
    def __init__(self, project_name="Exoplanet Habitability Analysis"):
        """Initialize project management system."""
        self.project_name = project_name
        self.tasks = pd.DataFrame(columns=[
            'task_id', 'task_name', 'description', 'assignee', 
            'start_date', 'end_date', 'status', 'dependencies', 
            'priority', 'module', 'completion_percentage'
        ])
        self.modules = [
            "Data Acquisition and Preparation",
            "Exploratory Data Analysis",
            "Habitable Zone Analysis",
            "Model Development",
            "Temporal Analysis",
            "Interactive Visualization Development",
            "Insights and Reporting",
            "Technical Implementation",
            "Project Management"
        ]
        self.status_types = ["Not Started", "In Progress", "Completed", "Blocked", "Delayed"]
        self.priority_levels = ["Low", "Medium", "High", "Critical"]
        
    def add_task(self, task_name, description, assignee, start_date, end_date, 
                 status="Not Started", dependencies=None, priority="Medium", 
                 module=None, completion_percentage=0):
        """Add a new task to the project."""
        if dependencies is None:
            dependencies = []
            
        # Validate inputs
        if status not in self.status_types:
            raise ValueError(f"Status must be one of {self.status_types}")
        if priority not in self.priority_levels:
            raise ValueError(f"Priority must be one of {self.priority_levels}")
        if module is not None and module not in self.modules:
            raise ValueError(f"Module must be one of {self.modules}")
        
        # Convert string dates to datetime if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d")
            
        # Generate task ID
        task_id = len(self.tasks) + 1
        
        # Create new task
        new_task = pd.DataFrame({
            'task_id': [task_id],
            'task_name': [task_name],
            'description': [description],
            'assignee': [assignee],
            'start_date': [start_date],
            'end_date': [end_date],
            'status': [status],
            'dependencies': [dependencies],
            'priority': [priority],
            'module': [module],
            'completion_percentage': [completion_percentage]
        })
        
        # Add to tasks dataframe
        self.tasks = pd.concat([self.tasks, new_task], ignore_index=True)
        return task_id
    
    def update_task_status(self, task_id, status, completion_percentage=None):
        """Update the status of a task."""
        if status not in self.status_types:
            raise ValueError(f"Status must be one of {self.status_types}")
            
        task_idx = self.tasks.index[self.tasks['task_id'] == task_id].tolist()
        if not task_idx:
            raise ValueError(f"Task ID {task_id} not found")
            
        self.tasks.at[task_idx[0], 'status'] = status
        
        if completion_percentage is not None:
            if not (0 <= completion_percentage <= 100):
                raise ValueError("Completion percentage must be between 0 and 100")
            self.tasks.at[task_idx[0], 'completion_percentage'] = completion_percentage
            
        # Auto-update completion percentage for completed tasks
        if status == "Completed" and completion_percentage is None:
            self.tasks.at[task_idx[0], 'completion_percentage'] = 100
    
    def get_project_timeline(self):
        """Generate project timeline data for visualization."""
        if self.tasks.empty:
            return pd.DataFrame()
        
        # Create a copy with properly formatted dates for display
        timeline_df = self.tasks.copy()
        return timeline_df
    
    def create_gantt_chart(self, interactive=True):
        """Create a Gantt chart of the project timeline."""
        if self.tasks.empty:
            print("No tasks to display")
            return None
            
        if interactive:
            # Create interactive Plotly Gantt chart
            df = self.tasks.copy()
            
            # Convert datetime objects to strings for plotly
            df['start_date_str'] = df['start_date'].dt.strftime('%Y-%m-%d')
            df['end_date_str'] = df['end_date'].dt.strftime('%Y-%m-%d')
            
            # Define colors based on status
            colors = {
                'Not Started': 'lightgrey',
                'In Progress': 'royalblue',
                'Completed': 'green',
                'Blocked': 'red',
                'Delayed': 'orange'
            }
            
            df['color'] = df['status'].map(colors)
            
            fig = ff.create_gantt(
                df, 
                colors=colors,
                index_col='task_id',
                show_colorbar=True,
                group_tasks=True,
                showgrid_x=True,
                showgrid_y=True,
                title=f'{self.project_name} - Project Timeline'
            )
            
            # Enhance the Gantt chart with more details
            for i, task in df.iterrows():
                fig.add_annotation(
                    x=task['end_date'],
                    y=task['task_id'],
                    text=f"{task['task_name']} ({task['completion_percentage']}%)",
                    showarrow=False,
                    font=dict(size=10)
                )
            
            fig.update_layout(
                autosize=True,
                height=max(500, len(df) * 30),
                margin=dict(l=150, r=20, t=50, b=100)
            )
            
            return fig
        else:
            # Create static matplotlib Gantt chart
            fig, ax = plt.subplots(figsize=(12, len(self.tasks) * 0.5 + 2))
            
            status_colors = {
                'Not Started': 'lightgrey',
                'In Progress': 'royalblue',
                'Completed': 'green',
                'Blocked': 'red',
                'Delayed': 'orange'
            }
            
            y_ticks = []
            y_labels = []
            
            for i, task in self.tasks.iterrows():
                start_date = task['start_date']
                end_date = task['end_date']
                duration = (end_date - start_date).days
                
                # Plot task bar
                ax.barh(
                    i, 
                    duration, 
                    left=mdates.date2num(start_date), 
                    color=status_colors[task['status']],
                    alpha=0.8,
                    height=0.5
                )
                
                # Add completion indicator if in progress
                if task['completion_percentage'] > 0 and task['completion_percentage'] < 100:
                    completed_width = duration * (task['completion_percentage'] / 100)
                    ax.barh(
                        i,
                        completed_width,
                        left=mdates.date2num(start_date),
                        color='darkgreen',
                        alpha=0.6,
                        height=0.5
                    )
                
                y_ticks.append(i)
                y_labels.append(f"{task['task_id']}. {task['task_name']} ({task['completion_percentage']}%)")
            
            # Format the x-axis
            ax.xaxis_date()
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Set y-axis
            ax.set_yticks(y_ticks)
            ax.set_yticklabels(y_labels)
            
            # Add grid lines
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color, label=status)
                for status, color in status_colors.items()
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.title(f'{self.project_name} - Project Timeline')
            plt.tight_layout()
            
            return fig
    
    def calculate_module_progress(self):
        """Calculate progress percentage for each module."""
        if self.tasks.empty:
            return pd.DataFrame(columns=['module', 'progress'])
            
        # Group by module and calculate average completion
        module_progress = self.tasks.groupby('module')['completion_percentage'].mean().reset_index()
        module_progress.columns = ['module', 'progress']
        
        # Fill in missing modules with 0 progress
        all_modules = pd.DataFrame({'module': self.modules})
        module_progress = pd.merge(all_modules, module_progress, on='module', how='left')
        module_progress['progress'] = module_progress['progress'].fillna(0)
        
        return module_progress
    
    def visualize_module_progress(self, interactive=True):
        """Create a bar chart showing progress of each module."""
        module_progress = self.calculate_module_progress()
        
        if interactive:
            fig = px.bar(
                module_progress,
                x='module',
                y='progress',
                title='Project Progress by Module',
                labels={'progress': 'Completion Percentage', 'module': 'Module'},
                color='progress',
                color_continuous_scale='viridis',
                range_color=[0, 100]
            )
            
            fig.update_layout(xaxis={'categoryorder': 'array', 'categoryarray': self.modules})
            
            # Add target line at 100%
            fig.add_shape(
                type="line",
                x0=-0.5,
                x1=len(module_progress) - 0.5,
                y0=100,
                y1=100,
                line=dict(color="red", width=2, dash="dash"),
            )
            
            return fig
        else:
            plt.figure(figsize=(12, 6))
            bars = plt.bar(module_progress['module'], module_progress['progress'])
            
            # Color bars based on progress
            for i, bar in enumerate(bars):
                progress = module_progress['progress'].iloc[i]
                if progress < 25:
                    bar.set_color('red')
                elif progress < 50:
                    bar.set_color('orange')
                elif progress < 75:
                    bar.set_color('yellow')
                else:
                    bar.set_color('green')
            
            plt.axhline(y=100, color='r', linestyle='--', alpha=0.7)
            plt.title('Project Progress by Module')
            plt.xlabel('Module')
            plt.ylabel('Completion Percentage')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            return plt.gcf()
    
    def generate_project_report(self):
        """Generate a comprehensive project status report."""
        if self.tasks.empty:
            return "No tasks available for report generation."
            
        # Calculate overall project progress
        overall_progress = self.tasks['completion_percentage'].mean()
        
        # Get tasks by status
        status_counts = self.tasks['status'].value_counts().to_dict()
        total_tasks = len(self.tasks)
        
        # Identify upcoming deadlines (next 7 days)
        today = datetime.now()
        upcoming = self.tasks[
            (self.tasks['end_date'] > today) & 
            (self.tasks['end_date'] <= today + timedelta(days=7)) &
            (self.tasks['status'] != 'Completed')
        ]
        
        # Identify overdue tasks
        overdue = self.tasks[
            (self.tasks['end_date'] < today) & 
            (self.tasks['status'] != 'Completed')
        ]
        
        # Generate module progress
        module_progress = self.calculate_module_progress()
        
        # Build report
        report = f"# {self.project_name} Status Report\n"
        report += f"**Generated on:** {today.strftime('%Y-%m-%d')}\n\n"
        
        report += "## Overall Progress\n"
        report += f"Project completion: {overall_progress:.1f}%\n\n"
        
        report += "## Task Status Summary\n"
        for status in self.status_types:
            count = status_counts.get(status, 0)
            percentage = (count / total_tasks * 100) if total_tasks > 0 else 0
            report += f"- {status}: {count} tasks ({percentage:.1f}%)\n"
        report += "\n"
        
        if not overdue.empty:
            report += "## Overdue Tasks\n"
            for _, task in overdue.iterrows():
                days_overdue = (today - task['end_date']).days
                report += f"- [{task['task_id']}] {task['task_name']} ({task['assignee']}) - {days_overdue} days overdue\n"
            report += "\n"
        
        if not upcoming.empty:
            report += "## Upcoming Deadlines (Next 7 Days)\n"
            for _, task in upcoming.iterrows():
                days_left = (task['end_date'] - today).days
                report += f"- [{task['task_id']}] {task['task_name']} ({task['assignee']}) - Due in {days_left} days\n"
            report += "\n"
        
        report += "## Module Progress\n"
        for _, row in module_progress.iterrows():
            report += f"- {row['module']}: {row['progress']:.1f}%\n"
        
        return report
    
    def save_project(self, filename="exoplanet_project.json"):
        """Save the project data to a JSON file."""
        # Convert the dataframe to a dictionary for serialization
        tasks_dict = self.tasks.copy()
        
        # Convert datetime objects to strings
        tasks_dict['start_date'] = tasks_dict['start_date'].dt.strftime('%Y-%m-%d')
        tasks_dict['end_date'] = tasks_dict['end_date'].dt.strftime('%Y-%m-%d')
        
        # Convert to dict for JSON serialization
        project_data = {
            "project_name": self.project_name,
            "tasks": tasks_dict.to_dict(orient='records'),
            "modules": self.modules,
            "status_types": self.status_types,
            "priority_levels": self.priority_levels
        }
        
        with open(filename, 'w') as f:
            json.dump(project_data, f, indent=2)
        
        print(f"Project saved to {filename}")
    
    def load_project(self, filename="exoplanet_project.json"):
        """Load project data from a JSON file."""
        if not os.path.exists(filename):
            print(f"File {filename} not found")
            return False
        
        with open(filename, 'r') as f:
            project_data = json.load(f)
        
        self.project_name = project_data.get("project_name", "Exoplanet Habitability Analysis")
        self.modules = project_data.get("modules", self.modules)
        self.status_types = project_data.get("status_types", self.status_types)
        self.priority_levels = project_data.get("priority_levels", self.priority_levels)
        
        # Convert tasks back to dataframe
        tasks_list = project_data.get("tasks", [])
        if tasks_list:
            self.tasks = pd.DataFrame(tasks_list)
            
            # Convert string dates back to datetime
            self.tasks['start_date'] = pd.to_datetime(self.tasks['start_date'])
            self.tasks['end_date'] = pd.to_datetime(self.tasks['end_date'])
        
        print(f"Project loaded from {filename}")
        return True
    
    def create_dependency_graph(self):
        """Create a network visualization of task dependencies."""
        if self.tasks.empty:
            print("No tasks to display")
            return None
        
        # Create nodes for each task
        nodes = []
        for _, task in self.tasks.iterrows():
            nodes.append({
                'id': task['task_id'],
                'label': f"{task['task_id']}. {task['task_name']}",
                'title': task['description'],
                'group': task['module'],
                'status': task['status']
            })
        
        # Create edges for dependencies
        edges = []
        for _, task in self.tasks.iterrows():
            dependencies = task['dependencies']
            if dependencies:
                for dep in dependencies:
                    edges.append({
                        'from': dep,
                        'to': task['task_id'],
                        'arrows': 'to'
                    })
        
        # Create network visualization using plotly
        node_x = []
        node_y = []
        node_text = []
        node_color = []
        
        # Simple layout for nodes - can be improved with more sophisticated algorithms
        for i, node in enumerate(nodes):
            angle = 2 * np.pi * i / len(nodes)
            radius = 10
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            
            node_x.append(x)
            node_y.append(y)
            node_text.append(node['label'])
            
            # Color by status
            status = node['status']
            if status == 'Completed':
                node_color.append('green')
            elif status == 'In Progress':
                node_color.append('blue')
            elif status == 'Blocked':
                node_color.append('red')
            elif status == 'Delayed':
                node_color.append('orange')
            else:
                node_color.append('lightgrey')
        
        # Create edges
        edge_x = []
        edge_y = []
        
        for edge in edges:
            from_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['from'])
            to_idx = next(i for i, node in enumerate(nodes) if node['id'] == edge['to'])
            
            edge_x.extend([node_x[from_idx], node_x[to_idx], None])
            edge_y.extend([node_y[from_idx], node_y[to_idx], None])
        
        # Create the graph
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines',
            name='Dependencies'
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition='bottom center',
            marker=dict(
                showscale=False,
                color=node_color,
                size=15,
                line_width=2,
                line=dict(color='black')
            ),
            hoverinfo='text',
            name='Tasks'
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{self.project_name} - Task Dependencies',
            titlefont_size=16,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            width=800,
            height=600
        )
        
        return fig

    def generate_burndown_chart(self, sprint_length_days=14):
        """Generate a burndown chart for the project or sprint."""
        if self.tasks.empty:
            print("No tasks to display")
            return None
        
        # Calculate total points/work remaining
        total_work = len(self.tasks) * 100  # Assuming each task is worth 100 points
        
        # Get date range for the project
        start_date = self.tasks['start_date'].min()
        end_date = self.tasks['end_date'].max()
        
        # If no dates are set, default to today and sprint_length_days
        if pd.isnull(start_date):
            start_date = datetime.now()
        if pd.isnull(end_date):
            end_date = start_date + timedelta(days=sprint_length_days)
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date)
        
        # Create ideal burndown - straight line from start to end
        ideal_burndown = pd.DataFrame({
            'date': date_range,
            'ideal_remaining': np.linspace(total_work, 0, len(date_range))
        })
        
        # Calculate actual burndown based on completion percentages
        # For each date, sum up the work remaining based on task completion
        actual_burndown = []
        
        for current_date in date_range:
            # For each task, calculate remaining work on this date
            remaining_work = 0
            for _, task in self.tasks.iterrows():
                # If the task has started by this date
                if task['start_date'] <= current_date:
                    # If the task is completed, no work remaining
                    if task['status'] == 'Completed' and task['end_date'] <= current_date:
                        remaining_work += 0
                    else:
                        # Calculate progress based on time or completion percentage
                        if task['completion_percentage'] > 0:
                            remaining_work += 100 - task['completion_percentage']
                        else:
                            # Assume linear progress between start and end dates
                            total_duration = (task['end_date'] - task['start_date']).days
                            if total_duration > 0:
                                elapsed = (current_date - task['start_date']).days
                                progress = min(1.0, max(0.0, elapsed / total_duration))
                                remaining_work += 100 * (1 - progress)
                            else:
                                remaining_work += 100  # Task with same start/end date
                else:
                    # Task hasn't started yet
                    remaining_work += 100
            
            actual_burndown.append({
                'date': current_date,
                'actual_remaining': remaining_work
            })
        
        actual_df = pd.DataFrame(actual_burndown)
        
        # Create interactive burndown chart
        fig = go.Figure()
        
        # Add ideal burndown line
        fig.add_trace(go.Scatter(
            x=ideal_burndown['date'],
            y=ideal_burndown['ideal_remaining'],
            mode='lines',
            name='Ideal Burndown',
            line=dict(color='blue', dash='dash')
        ))
        
        # Add actual burndown line
        fig.add_trace(go.Scatter(
            x=actual_df['date'],
            y=actual_df['actual_remaining'],
            mode='lines+markers',
            name='Actual Burndown',
            line=dict(color='red')
        ))
        
        # Update layout
        fig.update_layout(
            title=f'{self.project_name} - Burndown Chart',
            xaxis_title='Date',
            yaxis_title='Work Remaining',
            legend_title='Legend',
            hovermode='x unified',
            width=800,
            height=500
        )
        
        return fig


# Example usage
def create_sample_project():
    """Create a sample project with predefined tasks for demonstration purposes."""
    pm = ProjectManager("Exoplanet Habitability Analysis")
    
    # Add tasks for each module
    
    # Module 1: Data Acquisition and Preparation
    pm.add_task(
        task_name="Download NASA Exoplanet Archive data",
        description="Retrieve core dataset from NASA Exoplanet Archive API",
        assignee="Data Engineer",
        start_date="2025-05-20",
        end_date="2025-05-22",
        status="Not Started",
        dependencies=[],
        priority="High",
        module="Data Acquisition and Preparation",
        completion_percentage=0
    )
    
    pm.add_task(
        task_name="Clean and preprocess exoplanet data",
        description="Handle missing values, normalize units, and prepare features",
        assignee="Data Scientist",
        start_date="2025-05-22",
        end_date="2025-05-25",
        status="Not Started",
        dependencies=[1],
        priority="High",
        module="Data Acquisition and Preparation",
        completion_percentage=0
    )
    
    # Module 2: Exploratory Data Analysis
    pm.add_task(
        task_name="Perform statistical analysis of exoplanet features",
        description="Generate descriptive statistics and correlation analyses",
        assignee="Data Scientist",
        start_date="2025-05-25",
        end_date="2025-05-28",
        status="Not Started",
        dependencies=[2],
        priority="Medium",
        module="Exploratory Data Analysis",
        completion_percentage=0
    )
    
    # Module 3: Habitable Zone Analysis
    pm.add_task(
        task_name="Implement habitable zone calculation algorithms",
        description="Code the mathematical models for determining habitable zones",
        assignee="Astrophysicist",
        start_date="2025-05-28",
        end_date="2025-06-02",
        status="Not Started",
        dependencies=[3],
        priority="High",
        module="Habitable Zone Analysis",
        completion_percentage=0
    )
    
    # Module 4: Model Development
    pm.add_task(
        task_name="Develop habitability scoring model",
        description="Create algorithms to rate exoplanet habitability potential",
        assignee="ML Engineer",
        start_date="2025-06-02",
        end_date="2025-06-09",
        status="Not Started",
        dependencies=[4],
        priority="Critical",
        module="Model Development",
        completion_percentage=0
    )
    
    # Module 5: Temporal Analysis
    pm.add_task(
        task_name="Analyze exoplanet discovery trends",
        description="Study pattern of discoveries over time and by detection method",
        assignee="Data Analyst",
        start_date="2025-06-09",
        end_date="2025-06-14",
        status="Not Started",
        dependencies=[3],
        priority="Medium",
        module="Temporal Analysis",
        completion_percentage=0
    )
    
    # Module 6: Interactive Visualization Development
    pm.add_task(
        task_name="Create interactive dashboard",
        description="Build Plotly Dash application for exploring results",
        assignee="Visualization Developer",
        start_date="2025-06-14",
        end_date="2025-06-21",
        status="Not Started",
        dependencies=[5, 6],
        priority="High",
        module="Interactive Visualization Development",
        completion_percentage=0
    )
    
    # Module 7: Insights and Reporting
    pm.add_task(
        task_name="Generate final report with key findings",
        description="Compile results and conclusions from analysis",
        assignee="Project Lead",
        start_date="2025-06-21",
        end_date="2025-06-25",
        status="Not Started",
        dependencies=[7],
        priority="High",
        module="Insights and Reporting",
        completion_percentage=0
    )
    
    # Module 8: Technical Implementation
    pm.add_task(
        task_name="Deploy application to cloud server",
        description="Set up environment and deploy visualization app",
        assignee="DevOps Engineer",
        start_date="2025-06-25",
        end_date="2025-06-28",
        status="Not Started",
        dependencies=[7],
        priority="Medium",
        module="Technical Implementation",
        completion_percentage=0
    )
    
    # Module 9: Project Management
    pm.add_task(
        task_name="Weekly project status meetings",
        description="Coordinate team meetings and status updates",
        assignee="Project Manager",
        start_date="2025-05-20",
        end_date="2025-06-28",
        status="In Progress",
        dependencies=[],
        priority="Medium",
        module="Project Management",
        completion_percentage=10
    )
    
    return pm

# Demo code
if __name__ == "__main__":
    # Create a sample project
    project = create_sample_project()
    
    # Update some task statuses
    project.update_task_status(1, "Completed", 100)
    project.update_task_status(2, "In Progress", 75)
    project.update_task_status(3, "In Progress", 30)
    
    # Generate project visualizations
    gantt_chart = project.create_gantt_chart(interactive=True)
    module_progress = project.visualize_module_progress(interactive=True)
    dependency_graph = project.create_dependency_graph()
    burndown_chart = project.generate_burndown_chart()
    
    # Generate project report
    report = project.generate_project_report()
    print(report)
    
    # Save project data
    project.save_project("exoplanet_project_sample.json")