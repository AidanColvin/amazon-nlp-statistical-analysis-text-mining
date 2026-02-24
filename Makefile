.PHONY: setup run clean

setup:
	pip install scikit-learn xgboost scipy pandas numpy joblib

run: setup
	@echo "--- Starting Machine Learning Pipeline ---"
	python src/build_features.py
	python src/train_models.py

clean:
	rm -rf data/processed/*
	@echo "Cleaned processed data directory."