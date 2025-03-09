from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class AutoMLToolInput(BaseModel):
    """Input schema for AutoML Training Tool."""
    csv_path: str = Field(..., description="Path to the dataset CSV file.")
    target_column: str = Field(..., description="Target column name in the dataset.")
    model_type: str = Field("RandomForest", description="Type of model to train (default: RandomForest).")

class AutoMLTool(BaseTool):
    name: str = "AutoML Model Trainer"
    description: str = "Automatically trains a classification model with optimal hyperparameters."
    args_schema: Type[BaseModel] = AutoMLToolInput

    def _run(self, csv_path: str, target_column: str, model_type: str) -> str:
        """Executes AutoML training and saves the best model."""
        try:
            df = pd.read_csv(csv_path)
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Convert categorical variables
            X = pd.get_dummies(X)

            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_type == "RandomForest":
                model = RandomForestClassifier()
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            else:
                return "Currently, only RandomForest is supported."

            # Hyperparameter tuning
            grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
            grid_search.fit(X_train, y_train)

            # Get the best model
            best_model = grid_search.best_estimator_

            # Evaluate the model
            y_pred = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Save model
            model_filename = "best_model.pkl"
            joblib.dump(best_model, model_filename)

            return f"Training Complete ✅\nBest Parameters: {grid_search.best_params_}\nAccuracy: {accuracy:.4f}\nModel saved as {model_filename}"

        except Exception as e:
            return f"Error in training: {str(e)}"
