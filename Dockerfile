FROM public.ecr.aws/lambda/python:3.10

COPY src/requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY src/app.py ./app.py

COPY data/documents.json ./documents.json

CMD ["app.lambda_handler"]