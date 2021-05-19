online_inference

Сборка образа:
~~~
docker build -t yunusovev/online_inference:v2 .
~~~

Пулл из докерхаба:
~~~
docker pull yunusovev/online_inference:v2
~~~
Запуск образа:
~~~
docker run -p 8000:8000 yunusovev/online_inference:v2
~~~

Сделать запросы к контейнеру:
~~~
python -m ml_classifier_online.make_requests configs/requests_config.yaml
~~~

Уменьшение размера образа:
1) python 3.8 -> python 3.8-slim. Размер уменьшился с 1.57GB до 798MB
2) Не устанавливал дев зависимости в при сборке образа