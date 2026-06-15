FROM public.ecr.aws/lambda/python:3.12

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py preprocess.py inference.py schemas.py features.py ./
COPY model/ ./model/

CMD ["app.handler"]
