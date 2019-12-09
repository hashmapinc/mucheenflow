# Specify the base
FROM python:3.7.5-buster

# Create working directory
WORKDIR muhsheen

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy files over
ADD muhsheen .

RUN mkdir data && mkdir -p workflow/mlruns

ENV PYTHONPATH "${PYTHONPATH}:/"

# Execute command at runtime
CMD cd workflow && python run.py

