FROM continuumio/miniconda
COPY . .
WORKDIR /src
RUN conda env create -f environment_test.yml
RUN conda activate XCS224N