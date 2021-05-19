FROM python:3.8-slim
WORKDIR /app

COPY ruquirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY ml_project-1.0.0-py2.py3-none-any.whl ml_project-1.0.0-py2.py3-none-any.whl
RUN pip install ml_project-1.0.0-py2.py3-none-any.whl

COPY models/model_rf.pkl models/model_rf.pkl
COPY ml_classifier_online ./ml_classifier_online
COPY configs ./configs


CMD ["uvicorn", "ml_classifier_online:app", "--host", "0.0.0.0", "--port", "8000"]