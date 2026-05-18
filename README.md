# MIDIsynthesis - Генератор музыки на основе нейросети

![MIDIsynthesis Logo](https://via.placeholder.com/600x200/4a6fa5/ffffff?text=MIDIsynthesis+Music+Generator)

**MIDIsynthesis** - это проект для генерации музыки с использованием модели Transformer. Проект включает в себя:

- Модель для генерации MIDI музыки
- Веб-API на основе FastAPI
- Веб-интерфейс для удобной работы
- Кеширование результатов
- Полное логирование

## 📋 Содержание

1. [Требования](#requirements)
2. [Установка](#installation)
3. [Структура проекта](#project-structure)
4. [Использование через командную строку](#command-line-usage)
5. [Веб-API](#web-api)
6. [Веб-интерфейс](#web-interface)
7. [Кеширование](#caching)
8. [Логирование](#logging)
9. [Развертывание](#deployment)
10. [Примеры использования](#usage-examples)
11. [Решение проблем](#troubleshooting)
12. [Лицензия](#license)

## 🔧 Требования <a name="requirements"></a>

Для работы проекта необходимы:

- Python 3.11+
- PyTorch
- FastAPI
- Uvicorn
- music21
- Веб-браузер для интерфейса

## 🚀 Установка <a name="installation"></a>

### 1. Клонирование репозитория

```bash
git clone https://github.com/Bogdan-Fai/MIDIsynthesis.git
cd MIDIsynthesis
```

### 2. Установка зависимостей

```bash
uv sync
# Или если uv.lock нет, установите вручную:
pip install fastapi uvicorn python-multipart torch music21 mido pygame
```

### 3. Подготовка данных

Если отсутствует доступ в интернет, то убедитесь, что в папке `Data/outputs/` есть:

- `midi_transformer_final.pt` - обученная модель

```

## 📁 Структура проекта <a name="project-structure"></a>

```

MIDIsynthesis/
├── Data/
│ ├── outputs/ # Сгенерированные MIDI файлы и модели
│ └── vocab.json # Словарь токенов
├── Services/ # Вспомогательные сервисы
│ ├── midi_service.py # Работа с MIDI файлами
│ ├── miditxt_converter.py # Конвертация MIDI ↔ токены
│ └── tokenizer.py # Токенизация музыки
├── src/ # Исходный код
│ ├── dataset.py # Работа с dataset
│ ├── generate.py # Основная функция генерации
│ ├── model.py # Модель Transformer
│ ├── preprocess.py # Предобработка данных
│ └── train.py # Обучение модели
├── web/ # Веб-интерфейс
│ ├── static/ # Статические файлы
│ │ └── index.html # HTML интерфейс
│ └── main.py # FastAPI сервер
├── main.py # Основной скрипт (CLI)
├── README.md # Документация (этот файл)
├── USAGE.md # Инструкция по использованию
├── uv.lock # Зависимости
└── pyproject.toml # Конфигурация проекта

````

## 💻 Использование через командную строку <a name="command-line-usage"></a>

### Генерация музыки
```bash
# Базовая генерация
python main.py generate

# Генерация с seed (для воспроизводимости)
python main.py generate --seed 42

# Генерация с дополнительными параметрами
python main.py generate --seed 123 --temperature 0.8 --top_k 20
````

### Воспроизведение музыки

```bash
# Воспроизвести последний сгенерированный файл
python main.py last
```

### Просмотр сгенерированных файлов

```bash
# На Windows
dir Data\outputs\*.mid

# На Mac/Linux
ls Data/outputs/*.mid
```

## 🌐 Веб-API <a name="web-api"></a>

### Запуск сервера

```bash
# Запуск на порту 8000
python web/main.py

# Запуск на другом порту (например, 8003)
python -c "import uvicorn; uvicorn.run('web.main:app', host='0.0.0.0', port=8003)"
```

### Endpoints

#### 1. Генерация музыки

```
POST /api/generate?seed={seed}
```

**Параметры:**

- `seed` (опционально): Seed для воспроизводимости
- `temperature` (опционально): Температура генерации (0.1-2.0)
- `top_k` (опционально): Параметр top-k sampling

**Пример запроса:**

```bash
curl -X POST "http://localhost:8000/api/generate?seed=42"
```

**Пример ответа:**

```json
{
  "status": "success",
  "message": "Music generated successfully",
  "file_path": "Data/outputs/generated_20260516_170834.mid",
  "download_url": "/api/download/generated_20260516_170834.mid",
  "timestamp": "2026-05-16T17:08:34.460123",
  "from_cache": false
}
```

#### 2. Скачивание MIDI файла

```
GET /api/download/{filename}
```

**Пример запроса:**

```bash
curl -OJ "http://localhost:8000/api/download/generated_20260516_170834.mid"
```

#### 3. Документация API

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🎨 Веб-интерфейс <a name="web-interface"></a>

### Доступ к интерфейсу

Откройте в браузере:

```
http://localhost:8003/static/index.html
или
http://localhost:8000/static/index.html
```

### Возможности интерфейса

1. **Форма генерации**:
   - Ввод seed (опционально)
   - Настройка температуры (0.1-2.0)
   - Кнопка "Сгенерировать музыку"

2. **Результаты**:
   - Отображение информации о сгенерированном файле
   - Встроенный MIDI проигрыватель
   - Кнопка скачивания файла
   - Индикация кэшированных результатов

3. **История генераций**:
   - Сохранение до 10 последних генераций
   - Возможность загрузки параметров из истории
   - Отображение времени генерации

### Пример использования

1. Откройте интерфейс в браузере
2. Введите seed (например, 123) или оставьте пустым для случайного
3. Настройте температуру (по умолчанию 0.9)
4. Нажмите "Сгенерировать музыку"
5. Прослушайте результат прямо в браузере
6. Скачайте MIDI файл при необходимости

## 🗄️ Кеширование <a name="caching"></a>

### Как работает кэширование

- Результаты генерации кэшируются в памяти
- Время жизни кэша: 1 час (3600 секунд)
- Кэш работает по параметру seed
- При повторном запросе с тем же seed результат возвращается из кэша

### Преимущества кэширования

- Значительное ускорение повторных запросов
- Снижение нагрузки на сервер
- Экономия ресурсов

### Пример работы кэша

**Первый запрос (seed=111):**

```
2026-05-16 16:59:37,262 - INFO - Successfully generated music: Data/outputs\generated_20260516_165937.mid
2026-05-16 16:59:37,263 - INFO - Added to cache for seed: 111
```

**Повторный запрос (seed=111):**

```
2026-05-16 16:59:40,060 - INFO - Cache hit for seed: 111
2026-05-16 16:59:40,060 - INFO - Returning cached result for seed: 111
```

## 📝 Логирование <a name="logging"></a>

### Настройка логирования

Логи сохраняются в:

- Файл: `web/api.log`
- Консоль: stdout

### Уровни логирования

- `INFO`: Основные события (запуск сервера, генерация музыки)
- `ERROR`: Ошибки и исключения
- `WARNING`: Предупреждения

### Примеры логов

```
2026-05-16 16:57:49,177 - web.main - INFO - Starting MIDIsynthesis API server
2026-05-16 16:58:23,592 - web.main - INFO - Starting music generation with seed: 999
2026-05-16 16:58:34,460 - web.main - INFO - Successfully generated music: Data/outputs\generated_20260516_165834.mid
2026-05-16 16:58:34,460 - web.main - INFO - Added to cache for seed: 999
2026-05-16 16:59:40,060 - web.main - INFO - Cache hit for seed: 111
```

## 🎯 Примеры использования <a name="usage-examples"></a>

### Пример 1: Генерация музыки с разными seed

```bash
# Генерация с seed 1
python main.py generate --seed 1

# Генерация с seed 2
python main.py generate --seed 2

# Генерация с seed 3
python main.py generate --seed 3
```

### Пример 2: Использование API для интеграции

```python
import requests

# Генерация музыки
response = requests.post("http://localhost:8000/api/generate?seed=42")
data = response.json()

# Скачивание файла
midi_file = requests.get(f"http://localhost:8000{data['download_url']}")
with open("generated_music.mid", "wb") as f:
    f.write(midi_file.content)
```

### Пример 3: Использование веб-интерфейса

1. Откройте `http://localhost:8000/static/index.html`
2. Введите seed: 123
3. Настройте температуру: 0.8
4. Нажмите "Сгенерировать музыку"
5. Прослушайте результат в браузере
6. Скачайте файл при необходимости

## ⚠️ Решение проблем <a name="troubleshooting"></a>

### Ошибка: "No file 'Data/outputs/generated.mid' found"

**Проблема**: Команда `python main.py last` ищет файл с жестко закодированным именем.

**Решение**:

```bash
# Используйте конкретное имя файла
python -c "from Services.midi_service import play_midi; play_midi('Data/outputs/generated_20260516_165937.mid')"

# Или откройте файл вручную
start Data\outputs\generated_20260516_165937.mid
```

### Ошибка: "ModuleNotFoundError: No module named 'music21'"

**Проблема**: Не установлен пакет music21.

**Решение**:

```bash
pip install music21
```

### Ошибка: "FileNotFoundError: [Errno 2] No such file or directory: 'Data/outputs/midi_transformer_final_1.pt'"

**Проблема**: Отсутствует файл модели.

**Решение**:

```bash
# Скопируйте существующий файл
python -c "import shutil; shutil.copy('Data/outputs/midi_transformer_final.pt', 'Data/outputs/midi_transformer_final_1.pt')"
```

### Порт уже занят

**Проблема**: Нельзя запустить сервер на порту 8000.

**Решение**:

```bash
# Запустите на другом порту
python -c "import uvicorn; uvicorn.run('web.main:app', host='0.0.0.0', port=8001)"
```

## 📖 Дополнительная документация

- [USAGE.md](USAGE.md) - Подробная инструкция по использованию
- [FastAPI Documentation](https://fastapi.tiangolo.com/) - Документация FastAPI
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) - Документация PyTorch
- [music21 Documentation](https://web.mit.edu/music21/) - Документация music21

## 🔄 Обновления

### Версия 0.1.0 (Текущая)

- Базовая генерация музыки
- Веб-API с FastAPI
- Веб-интерфейс
- Кеширование результатов
- Полное логирование

### Планы на будущее

- Добавление аутентификации
- Интеграция с Redis для распределенного кэша
- Docker контейнеризация
- CI/CD пайплайны
- Расширенная документация

## 📄 Лицензия <a name="license"></a>

Этот проект лицензируется по лицензии MIT. Подробности смотрите в файле LICENSE.

---

**MIDIsynthesis** © 2026. Все права защищены.
