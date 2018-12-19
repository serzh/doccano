FROM python:3.6

WORKDIR /opt/doccano

COPY requirements.txt .
RUN pip install -q -r requirements.txt

COPY . .

CMD /usr/local/bin/gunicorn -b 0.0.0.0:8000 --pythonpath app app.wsgi