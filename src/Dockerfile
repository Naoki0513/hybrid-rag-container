FROM public.ecr.aws/lambda/python:3.10

COPY requirements.txt ./

RUN pip install -r requirements.txt

COPY src/app.py ./app.py
COPY data/documents.json ./

RUN rm ./documents.json

CMD ["app.lambda_handler"]