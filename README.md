# telegram_bot
Сборка образа
docker build -t telegram_bot .

Запуск контейнера с удалением после остановки
docker run --rm telegram_bot

Список остановленных контейнеров
docker ps -a -q

Удаление всех остановленных контейнеров
docker rm $(docker ps -qa)

Запуск контейнера с volume
docker run --rm --name telegram_bot -v bot:/usr/src/app/ telegram_bot
