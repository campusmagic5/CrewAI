from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from auto_ml.tools.custom_tool import AutoMLTool
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from crewai.tools import tool

#Tool using @tool decorator
# @tool("automl_model_trainer")
# def automl_model_trainer(csv_path: str, target_column: str, model_type: str) -> str:
#     """
#     Trains a classification model with hyperparameter tuning using GridSearchCV.
    
#     Args:
#         csv_path (str): Path to the CSV dataset.
#         target_column (str): Name of the target column.
#         model_type (str): Model to use ('RandomForest' or 'XGBoost'). Default: RandomForest.
    
#     Returns:
#         str: Success message with best parameters and model path.
#     """
#     try:
#         # Load dataset
#         df = pd.read_csv(csv_path)
        
#         # Split into features and target
#         X = df.drop(columns=[target_column])
#         y = df[target_column]
        
#         # Handle categorical data (if any)
#         X = pd.get_dummies(X)

#         # Train-Test Split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Select model type
#         if model_type == "RandomForest":
#             model = RandomForestClassifier()
#             param_grid = {
#                 'n_estimators': [50, 100, 200],
#                 'max_depth': [None, 10, 20],
#                 'min_samples_split': [2, 5, 10]
#             }
#         else:
#             return "Currently, only RandomForest model is supported."

#         # Hyperparameter tuning
#         grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
#         grid_search.fit(X_train, y_train)

#         # Get the best model
#         best_model = grid_search.best_estimator_

#         # Evaluate on test data
#         y_pred = best_model.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)

#         # Save the best model
#         model_filename = "best_model.pkl"
#         joblib.dump(best_model, model_filename)

#         return f"AutoML Training Complete ✅\nBest Parameters: {grid_search.best_params_}\nAccuracy: {accuracy:.4f}\nModel saved as {model_filename}"

#     except Exception as e:
#         return f"Error in AutoML Training: {str(e)}"

# If you want to run a snippet of code before or after the crew starts, 
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AutoMl():
	"""AutoMl crew"""

	# Learn more about YAML configuration files here:
	# Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
	# Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended
	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	# If you would like to add tools to your agents, you can learn more about it here:
	# https://docs.crewai.com/concepts/agents#agent-tools
	@agent
	def AutoML_Engineer(self) -> Agent:
		return Agent(
			config=self.agents_config['AutoML_Engineer'],
			verbose=True,
			tools= [AutoMLTool]
		)

	# @agent
	# def reporting_analyst(self) -> Agent:
	# 	return Agent(
	# 		config=self.agents_config['reporting_analyst'],
	# 		verbose=True
	# 	)

	# To learn more about structured task outputs, 
	# task dependencies, and task callbacks, check out the documentation:
	# https://docs.crewai.com/concepts/tasks#overview-of-a-task
	@task
	def ml_task(self) -> Task:
		return Task(
			config=self.tasks_config['ml_task'],
		)

	# @task
	# def reporting_task(self) -> Task:
	# 	return Task(
	# 		config=self.tasks_config['reporting_task'],
	# 		output_file='report.md'
	# 	)

	@crew
	def crew(self) -> Crew:
		"""Creates the AutoMl crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)
