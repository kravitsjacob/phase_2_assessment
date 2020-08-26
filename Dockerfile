FROM python
RUN pip3 install pandas
RUN pip3 install numpy
RUN pip3 install scikit-learn
RUN pip3 install imbalanced-learn
RUN pip3 install tables
WORKDIR /app
COPY ["assess.py", "/app/"]
ENTRYPOINT ["python", "-u", "assess.py"]

