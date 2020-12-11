FROM python:3.7

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose port 
EXPOSE 8080

# Run the application:
CMD ["gunicorn"  , "-b", "0.0.0.0:8080", "app:app"]
