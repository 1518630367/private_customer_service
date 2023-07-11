FROM continuumio/anaconda3
#FROM python
ADD ./ /customer_service
WORKDIR /customer_service
COPY . .
RUN pip install -i https://pypi.douban.com/simple gunicorn gevent
RUN pip install -i https://pypi.douban.com/simple -r requirements.txt
ENTRYPOINT python /customer_service/customer\ service.py

#ENTRYPOINT nohup python only_AE.py >> output.log 2>&1 &
