"""
Model Development Module

This module contains functions for building, training, and evaluating machine learning
models for exoplanet analysis and habitability prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import joblib
import logging

# Configure logger
logger = logging.getLogger(__name__)

class ExoplanetModel:
    """Class for exoplanet habitability prediction models."""
    
    def __init__(self, model_type: str = 'random_forest', model_params: Dict = None):
        """
        Initialize the exoplanet model.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'svm')
            model_params: Dictionary of parameters to use for the model
        """
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_name = None
        self.numeric_features = None
        self.categorical_features = None
        
        logger.info(f"Initialized {model_type} model")
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a data preprocessor for feature transformation.
        
        Args:
            X: Feature dataframe
            
        Returns:
            sklearn ColumnTransformer for preprocessing
        """
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        logger.debug(f"Numeric features: {numeric_features}")
        logger.debug(f"Categorical features: {categorical_features}")
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessing for categorical features if they exist
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ]
            )
        
        return preprocessor
    
    def _create_model(self) -> Any:
        """
        Create the machine learning model based on model_type.
        
        Returns:
            sklearn model instance
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                max_features=self.model_params.get('max_features', 'sqrt'),
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                C=self.model_params.get('C', 1.0),
                kernel=self.model_params.get('kernel', 'rbf'),
                gamma=self.model_params.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target: str, 
        features: List[str] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training by splitting into train/test sets.
        
        Args:
            df: DataFrame containing features and target
            target: Name of the target column
            features: List of feature columns to use (if None, all columns except target are used)
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # If features not specified, use all columns except target
        if features is None:
            features = [col for col in df.columns if col != target]
        
        logger.info(f"Preparing data with {len(features)} features: {features[:5]}{'...' if len(features) > 5 else ''}")
        
        # Store feature and target names
        self.feature_names = features
        self.target_name = target
        
        # Check if target exists in dataframe
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        
        # Check if all features exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataframe: {missing_features}")
        
        # Split data into features and target
        X = df[features]
        y = df[target]
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame = None, 
        y_test: pd.Series = None,
        tune_hyperparameters: bool = False
    ) -> Dict:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional, for immediate evaluation)
            y_test: Test target (optional, for immediate evaluation)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X_train)
        
        # Create machine learning model
        base_model = self._create_model()
        
        if tune_hyperparameters:
            logger.info(f"Performing hyperparameter tuning for {self.model_type}")
            
            # Define parameter grid based on model type
            if self.model_type == 'random_forest':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'model__C': [0.1, 1, 10, 100],
                    'model__gamma': ['scale', 'auto', 0.1, 0.01],
                    'model__kernel': ['rbf', 'linear', 'poly']
                }
            
            # Create full pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', base_model)
            ])
            
            # Create grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            self.model = grid_search.best_estimator_
            best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
            
            logger.info(f"Best parameters: {best_params}")
            
            # Update model parameters
            self.model_params.update(best_params)
            
            # Results to return
            results = {
                'best_params': best_params,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        else:
            # Create and train pipeline without hyperparameter tuning
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', base_model)
            ])
            
            logger.info(f"Training {self.model_type} model on {X_train.shape[0]} samples")
            self.model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
            
            results = {
                'cv_mean_score': cv_scores.mean(),
                'cv_std_score': cv_scores.std()
            }
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            test_results = self.evaluate(X_test, y_test)
            results.update(test_results)
        
        return results
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating {self.model_type} model on {X_test.shape[0]} samples")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        cls_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision_score': avg_precision,
            'confusion_matrix': cm,
            'classification_report': cls_report,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr
            },
            'pr_curve': {
                'precision': precision_curve,
                'recall': recall_curve
            }
        }
        
        logger.info(f"Evaluation results: accuracy={accuracy:.4f}, precision={precision:.4f}, "
                   f"recall={recall:.4f}, f1={f1:.4f}, ROC AUC={roc_auc:.4f}")
        
        return results
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Calculate feature importance for the trained model.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary with feature importance results
        """
        if self.model is None:
            raise ValueError("Model must be trained before calculating feature importance")
        
        logger.info("Calculating feature importance")
        
        # For Random Forest, extract feature importance directly
        if self.model_type == 'random_forest':
            # Get the actual model from the pipeline
            rf_model = self.model.named_steps['model']
            
            # Get feature importance
            importances = rf_model.feature_importances_
            
            # Get feature names from preprocessor
            feature_names = []
            
            # Extract feature names from preprocessor
            # This is more complex due to one-hot encoding of categorical features
            if self.categorical_features:
                # For one-hot encoded features, we need to get the original feature names
                ct = self.model.named_steps['preprocessor']
                
                # Get the names of numeric features
                numeric_features = self.numeric_features
                
                # Get the names of categorical features after one-hot encoding
                ohe = ct.named_transformers_['cat'].named_steps['onehot']
                categorical_features_encoded = ohe.get_feature_names_out(self.categorical_features)
                
                # Combine all feature names
                feature_names = np.append(numeric_features, categorical_features_encoded)
            else:
                feature_names = self.numeric_features
            
            # Create sorted feature importance
            feature_importance = {}
            for feature, importance in zip(feature_names, importances):
                feature_importance[feature] = importance
            
            # Sort by importance
            feature_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
            
            results = {
                'feature_importance': feature_importance,
                'method': 'built_in'
            }
        else:
            # For other models, use permutation importance
            # This is more computationally expensive but works for any model
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42
            )
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(X.columns):
                feature_importance[feature] = perm_importance.importances_mean[i]
            
            # Sort by importance
            feature_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
            
            results = {
                'feature_importance': feature_importance,
                'method': 'permutation'
            }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature data
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making probability predictions on {X.shape[0]} samples")
        
        # Make probability predictions
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to a file.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")
        
        logger.info(f"Saving model to {filepath}")
        
        # Create model info dictionary
        model_info = {
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        
        # Save model
        joblib.dump(model_info, filepath)
    
    @classmethod
    def load_model(cls, filepath: str) -> 'ExoplanetModel':
        """
        Load a trained model from a file.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded ExoplanetModel instance
        """
        logger.info(f"Loading model from {filepath}")
        
        # Load model info
        model_info = joblib.load(filepath)
        
        # Create new model instance
        model = cls(model_type=model_info['model_type'], model_params=model_info['model_params'])
        
        # Set model attributes
        model.model = model_info['model']
        model.feature_names = model_info['feature_names']
        model.target_name = model_info['target_name']
        model.numeric_features = model_info['numeric_features']
        model.categorical_features = model_info['categorical_features']
        
        return model

def plot_roc_curve(results: Dict, title: str = 'ROC Curve', save_path: Optional[str] = None) -> None:
    """
    Plot the ROC curve from model evaluation results.
    
    Args:
        results: Dictionary with model evaluation results
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'roc_curve' not in results:
        logger.error("ROC curve data not found in results")
        return
    
    fpr = results['roc_curve']['fpr']
    tpr = results['roc_curve']['tpr']
    roc_auc = results['roc_auc']
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def plot_precision_recall_curve(results: Dict, title: str = 'Precision-Recall Curve', save_path: Optional[str] = None) -> None:
    """
    Plot the precision-recall curve from model evaluation results.
    
    Args:
        results: Dictionary with model evaluation results
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'pr_curve' not in results:
        logger.error("Precision-recall curve data not found in results")
        return
    
    precision = results['pr_curve']['precision']
    recall = results['pr_curve']['recall']
    avg_precision = results['avg_precision_score']
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'P-R curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def plot_confusion_matrix(results: Dict, title: str = 'Confusion Matrix', save_path: Optional[str] = None) -> None:
    """
    Plot the confusion matrix from model evaluation results.
    
    Args:
        results: Dictionary with model evaluation results
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'confusion_matrix' not in results:
        logger.error("Confusion matrix not found in results")
        return
    
    cm = results['confusion_matrix']
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def plot_feature_importance(features_results: Dict, top_n: int = 20, title: str = 'Feature Importance', 
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance from feature importance results.
    
    Args:
        features_results: Dictionary with feature importance results
        top_n: Number of top features to plot
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'feature_importance' not in features_results:
        logger.error("Feature importance not found in results")
        return
    
    feature_importance = features_results['feature_importance']
    
    # Get top N features
    top_features = list(feature_importance.keys())[:top_n]
    top_importance = [feature_importance[f] for f in top_features]
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_importance, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def compare_models(models_results: Dict, metric: str = 'f1_score', title: str = 'Model Comparison', 
                 save_path: Optional[str] = None) -> None:
    """
    Compare multiple models based on a specific metric.
    
    Args:
        models_results: Dictionary with model evaluation results for multiple models
        metric: Metric to compare models by
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if not models_results:
        logger.error("No model results provided")
        return
    
    # Check if metric exists in all model results
    for model_name, results in models_results.items():
        if metric not in results:
            logger.error(f"Metric '{metric}' not found in results for model '{model_name}'")
            return
    
    # Extract metrics for each model
    model_names = list(models_results.keys())
    metric_values = [models_results[model][metric] for model in model_names]
    
    plt.figure(figsize=(10, 8))
    plt.bar(model_names, metric_values)
    plt.ylim([0, 1.05])
    plt.xlabel('Model')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, value in enumerate(metric_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    else:
        plt.show()
        
    plt.close()
        
"""
Model Development Module

This module contains functions for building, training, and evaluating machine learning
models for exoplanet analysis and habitability prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.inspection import permutation_importance
import joblib
import logging

# Configure logger
logger = logging.getLogger(__name__)

class ExoplanetModel:
    """Class for exoplanet habitability prediction models."""
    
    def __init__(self, model_type: str = 'random_forest', model_params: Dict = None):
        """
        Initialize the exoplanet model.
        
        Args:
            model_type: Type of model to use ('random_forest' or 'svm')
            model_params: Dictionary of parameters to use for the model
        """
        self.model_type = model_type.lower()
        self.model_params = model_params or {}
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_name = None
        self.numeric_features = None
        self.categorical_features = None
        
        logger.info(f"Initialized {model_type} model")
    
    def _create_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Create a data preprocessor for feature transformation.
        
        Args:
            X: Feature dataframe
            
        Returns:
            sklearn ColumnTransformer for preprocessing
        """
        # Identify numeric and categorical features
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        logger.debug(f"Numeric features: {numeric_features}")
        logger.debug(f"Categorical features: {categorical_features}")
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Create preprocessing for categorical features if they exist
        if categorical_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features),
                    ('cat', categorical_transformer, categorical_features)
                ]
            )
        else:
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numeric_features)
                ]
            )
        
        return preprocessor
    
    def _create_model(self) -> Any:
        """
        Create the machine learning model based on model_type.
        
        Returns:
            sklearn model instance
        """
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=self.model_params.get('n_estimators', 100),
                max_depth=self.model_params.get('max_depth', None),
                min_samples_split=self.model_params.get('min_samples_split', 2),
                min_samples_leaf=self.model_params.get('min_samples_leaf', 1),
                max_features=self.model_params.get('max_features', 'sqrt'),
                random_state=42
            )
        elif self.model_type == 'svm':
            return SVC(
                C=self.model_params.get('C', 1.0),
                kernel=self.model_params.get('kernel', 'rbf'),
                gamma=self.model_params.get('gamma', 'scale'),
                probability=True,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def prepare_data(
        self, 
        df: pd.DataFrame, 
        target: str, 
        features: List[str] = None,
        test_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for model training by splitting into train/test sets.
        
        Args:
            df: DataFrame containing features and target
            target: Name of the target column
            features: List of feature columns to use (if None, all columns except target are used)
            test_size: Proportion of data to use for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # If features not specified, use all columns except target
        if features is None:
            features = [col for col in df.columns if col != target]
        
        logger.info(f"Preparing data with {len(features)} features: {features[:5]}{'...' if len(features) > 5 else ''}")
        
        # Store feature and target names
        self.feature_names = features
        self.target_name = target
        
        # Check if target exists in dataframe
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found in dataframe")
        
        # Check if all features exist in dataframe
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataframe: {missing_features}")
        
        # Split data into features and target
        X = df[features]
        y = df[target]
        
        # Create train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        logger.info(f"Data split: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        return X_train, X_test, y_train, y_test
    
    def train(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_test: pd.DataFrame = None, 
        y_test: pd.Series = None,
        tune_hyperparameters: bool = False
    ) -> Dict:
        """
        Train the model on the provided data.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features (optional, for immediate evaluation)
            y_test: Test target (optional, for immediate evaluation)
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        # Create preprocessor
        self.preprocessor = self._create_preprocessor(X_train)
        
        # Create machine learning model
        base_model = self._create_model()
        
        if tune_hyperparameters:
            logger.info(f"Performing hyperparameter tuning for {self.model_type}")
            
            # Define parameter grid based on model type
            if self.model_type == 'random_forest':
                param_grid = {
                    'model__n_estimators': [50, 100, 200],
                    'model__max_depth': [None, 10, 20, 30],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4]
                }
            elif self.model_type == 'svm':
                param_grid = {
                    'model__C': [0.1, 1, 10, 100],
                    'model__gamma': ['scale', 'auto', 0.1, 0.01],
                    'model__kernel': ['rbf', 'linear', 'poly']
                }
            
            # Create full pipeline
            pipeline = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', base_model)
            ])
            
            # Create grid search
            grid_search = GridSearchCV(
                pipeline,
                param_grid=param_grid,
                cv=5,
                scoring='f1',
                n_jobs=-1
            )
            
            # Fit grid search
            grid_search.fit(X_train, y_train)
            
            # Get best model and parameters
            self.model = grid_search.best_estimator_
            best_params = {k.replace('model__', ''): v for k, v in grid_search.best_params_.items()}
            
            logger.info(f"Best parameters: {best_params}")
            
            # Update model parameters
            self.model_params.update(best_params)
            
            # Results to return
            results = {
                'best_params': best_params,
                'best_score': grid_search.best_score_,
                'cv_results': grid_search.cv_results_
            }
            
        else:
            # Create and train pipeline without hyperparameter tuning
            self.model = Pipeline(steps=[
                ('preprocessor', self.preprocessor),
                ('model', base_model)
            ])
            
            logger.info(f"Training {self.model_type} model on {X_train.shape[0]} samples")
            self.model.fit(X_train, y_train)
            
            # Cross-validation score
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='f1')
            
            results = {
                'cv_mean_score': cv_scores.mean(),
                'cv_std_score': cv_scores.std()
            }
        
        # Evaluate on test set if provided
        if X_test is not None and y_test is not None:
            test_results = self.evaluate(X_test, y_test)
            results.update(test_results)
        
        return results
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the trained model on test data.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        
        logger.info(f"Evaluating {self.model_type} model on {X_test.shape[0]} samples")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Calculate ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        # Calculate precision-recall curve
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)
        avg_precision = average_precision_score(y_test, y_prob)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        cls_report = classification_report(y_test, y_pred, output_dict=True)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision_score': avg_precision,
            'confusion_matrix': cm,
            'classification_report': cls_report,
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr
            },
            'pr_curve': {
                'precision': precision_curve,
                'recall': recall_curve
            }
        }
        
        logger.info(f"Evaluation results: accuracy={accuracy:.4f}, precision={precision:.4f}, "
                   f"recall={recall:.4f}, f1={f1:.4f}, ROC AUC={roc_auc:.4f}")
        
        return results
    
    def get_feature_importance(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Calculate feature importance for the trained model.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary with feature importance results
        """
        if self.model is None:
            raise ValueError("Model must be trained before calculating feature importance")
        
        logger.info("Calculating feature importance")
        
        # For Random Forest, extract feature importance directly
        if self.model_type == 'random_forest':
            # Get the actual model from the pipeline
            rf_model = self.model.named_steps['model']
            
            # Get feature importance
            importances = rf_model.feature_importances_
            
            # Get feature names from preprocessor
            feature_names = []
            
            # Extract feature names from preprocessor
            # This is more complex due to one-hot encoding of categorical features
            if self.categorical_features:
                # For one-hot encoded features, we need to get the original feature names
                ct = self.model.named_steps['preprocessor']
                
                # Get the names of numeric features
                numeric_features = self.numeric_features
                
                # Get the names of categorical features after one-hot encoding
                ohe = ct.named_transformers_['cat'].named_steps['onehot']
                categorical_features_encoded = ohe.get_feature_names_out(self.categorical_features)
                
                # Combine all feature names
                feature_names = np.append(numeric_features, categorical_features_encoded)
            else:
                feature_names = self.numeric_features
            
            # Create sorted feature importance
            feature_importance = {}
            for feature, importance in zip(feature_names, importances):
                feature_importance[feature] = importance
            
            # Sort by importance
            feature_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
            
            results = {
                'feature_importance': feature_importance,
                'method': 'built_in'
            }
        else:
            # For other models, use permutation importance
            # This is more computationally expensive but works for any model
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                self.model, X, y, n_repeats=10, random_state=42
            )
            
            # Create feature importance dictionary
            feature_importance = {}
            for i, feature in enumerate(X.columns):
                feature_importance[feature] = perm_importance.importances_mean[i]
            
            # Sort by importance
            feature_importance = {k: v for k, v in sorted(
                feature_importance.items(), key=lambda item: item[1], reverse=True
            )}
            
            results = {
                'feature_importance': feature_importance,
                'method': 'permutation'
            }
        
        return results
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Feature data
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        logger.info(f"Making probability predictions on {X.shape[0]} samples")
        
        # Make probability predictions
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str = "C:/Users/Yatharth Vashisht/Desktop/exoplanet exploratory data anaylysis/Models/exoplanet_model.pkl") -> None:
        """
        Save the trained model to a file.

        Args:
            filepath: Path to save the model (default is the Models folder on Desktop)
        """
        if self.model is None:
            raise ValueError("Model must be trained before saving")

        logger.info(f"Saving model to {filepath}")

        # Create model info dictionary
        model_info = {
            'model': self.model,
            'model_type': self.model_type,
            'model_params': self.model_params,
            'feature_names': self.feature_names,
            'target_name': self.target_name,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }

        # Save model
        joblib.dump(model_info, filepath)
    
    @classmethod
    def load_model(cls, filepath: str = "C:/Users/Yatharth Vashisht/Desktop/exoplanet exploratory data anaylysis/Models/exoplanet_model.pkl") -> 'ExoplanetModel':
        """
        Load a trained model from a file.

        Args:
            filepath: Path to the saved model (default is the Models folder on Desktop)

        Returns:
            Loaded ExoplanetModel instance
        """
       
        logger.info(f"Loading model from {filepath}")

        # Load model info
        model_info = joblib.load(filepath)

        # Create new model instance
        model = cls(model_type=model_info['model_type'], model_params=model_info['model_params'])

        # Set model attributes
        model.model = model_info['model']
        model.feature_names = model_info['feature_names']
        model.target_name = model_info['target_name']
        model.numeric_features = model_info['numeric_features']
        model.categorical_features = model_info['categorical_features']

        return model

        return model

def plot_roc_curve(results: Dict, title: str = 'ROC Curve', save_path: Optional[str] = None) -> None:
    """
    Plot the ROC curve from model evaluation results.
    
    Args:
        results: Dictionary with model evaluation results
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'roc_curve' not in results:
        logger.error("ROC curve data not found in results")
        return
    
    fpr = results['roc_curve']['fpr']
    tpr = results['roc_curve']['tpr']
    roc_auc = results['roc_auc']
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved ROC curve plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def plot_precision_recall_curve(results: Dict, title: str = 'Precision-Recall Curve', save_path: Optional[str] = None) -> None:
    """
    Plot the precision-recall curve from model evaluation results.
    
    Args:
        results: Dictionary with model evaluation results
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'pr_curve' not in results:
        logger.error("Precision-recall curve data not found in results")
        return
    
    precision = results['pr_curve']['precision']
    recall = results['pr_curve']['recall']
    avg_precision = results['avg_precision_score']
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, label=f'P-R curve (AP = {avg_precision:.3f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(title)
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved precision-recall curve plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def plot_confusion_matrix(results: Dict, title: str = 'Confusion Matrix', save_path: Optional[str] = None) -> None:
    """
    Plot the confusion matrix from model evaluation results.
    
    Args:
        results: Dictionary with model evaluation results
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'confusion_matrix' not in results:
        logger.error("Confusion matrix not found in results")
        return
    
    cm = results['confusion_matrix']
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved confusion matrix plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def plot_feature_importance(features_results: Dict, top_n: int = 20, title: str = 'Feature Importance', 
                          save_path: Optional[str] = None) -> None:
    """
    Plot feature importance from feature importance results.
    
    Args:
        features_results: Dictionary with feature importance results
        top_n: Number of top features to plot
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if 'feature_importance' not in features_results:
        logger.error("Feature importance not found in results")
        return
    
    feature_importance = features_results['feature_importance']
    
    # Get top N features
    top_features = list(feature_importance.keys())[:top_n]
    top_importance = [feature_importance[f] for f in top_features]
    
    plt.figure(figsize=(12, 10))
    plt.barh(range(len(top_features)), top_importance, align='center')
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved feature importance plot to {save_path}")
    else:
        plt.show()
        
    plt.close()

def compare_models(models_results: Dict, metric: str = 'f1_score', title: str = 'Model Comparison', 
                 save_path: Optional[str] = None) -> None:
    """
    Compare multiple models based on a specific metric.
    
    Args:
        models_results: Dictionary with model evaluation results for multiple models
        metric: Metric to compare models by
        title: Title for the plot
        save_path: Path to save the figure (if None, the figure is displayed but not saved)
    """
    if not models_results:
        logger.error("No model results provided")
        return
    
    # Check if metric exists in all model results
    for model_name, results in models_results.items():
        if metric not in results:
            logger.error(f"Metric '{metric}' not found in results for model '{model_name}'")
            return
    
    # Extract metrics for each model
    model_names = list(models_results.keys())
    metric_values = [models_results[model][metric] for model in model_names]
    
    plt.figure(figsize=(10, 8))
    plt.bar(model_names, metric_values)
    plt.ylim([0, 1.05])
    plt.xlabel('Model')
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add values on top of bars
    for i, value in enumerate(metric_values):
        plt.text(i, value + 0.01, f'{value:.3f}', ha='center')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved model comparison plot to {save_path}")
    else:
        plt.show()
        
    plt.close()