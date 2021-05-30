lint:
	@mypy ml_classifier
	@mypy tests
	@flake8 ml_classifier

train:
	@python -m ml_classifier.trainer ${config}

predict:
	@python -m ml_classifier.predict ${config}

