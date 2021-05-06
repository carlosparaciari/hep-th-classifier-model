# Define global args
ARG RUNTIME_VERSION="3.7"

# STAGE 1 - Build base image with AWS runtime
FROM public.ecr.aws/lambda/python:${RUNTIME_VERSION}

# STAGE 2 - Create function directory and dependences
ARG RUNTIME_VERSION

# Copy handler function
COPY app/* ./

# Install the function's dependencies
RUN python${RUNTIME_VERSION} -m pip install -r requirements.txt --target ./

# Install nltk packages
RUN python -m nltk.downloader -d ./nltk_data stopwords wordnet

# STAGE 3 - Final runtime image
CMD [ "lambda_classify.handler" ]