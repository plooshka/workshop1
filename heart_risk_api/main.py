import pandas as pd

import joblib

import io

import lightgbm as lgb

import json

from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.templating import Jinja2Templates 
from fastapi.responses import HTMLResponse
from fastapi.requests import Request

import numpy as np

# --- 1. Инициализация приложения FastAPI ---
app = FastAPI(
    title="API для предсказания риска сердечного приступа",
    description="Загрузите CSV файл с данными пациентов для получения предсказаний риска сердечного приступа (вероятность от 0 до 1)."
)

# Указываем FastAPI, где искать шаблоны
templates = Jinja2Templates(directory="templates")

# --- 2. Загрузка модели при старте приложения ---
# Модель загружается ОДИН РАЗ при запуске приложения, а не при каждом запросе.
try:
    model_filename = 'lgbm_final_model.joblib'
    model = joblib.load(model_filename)
    # Предполагаем, что модель LightGBM имеет атрибут feature_name_
    # Если нет, вам нужно будет вручную указать порядок признаков,
    # на котором обучалась модель.
    model_features = model.feature_name_
    print(f"✅ Модель '{model_filename}' успешно загружена.")
except FileNotFoundError:
    raise RuntimeError(f"❌ Файл модели '{model_filename}' не найден. Убедитесь, что он находится в той же папке.")
except Exception as e:
    raise RuntimeError(f"❌ Ошибка при загрузке модели: {e}")

FEATURE_DESCRIPTIONS = {
    "age": "Возраст (нормализованное число от 0 до 1)",
    "cholesterol": "Холестерин (нормализованное число от 0 до 1)",
    "heart_rate": "Частота сердечных сокращений",
    "diabetes": "Наличие диабета (1 - да, 0 - нет)",
    "family_history": "Семейная история болезней (1 - да, 0 - нет)",
    "smoking": "Курение (1 - да, 0 - нет)",
    "obesity": "Ожирение (1 - да, 0 - нет)",
    "alcohol_consumption": "Потребление алкоголя (1 - да, 0 - нет)",
    "exercise_hours_per_week": "Часы упражнений в неделю",
    "diet": "Тип диеты (0 - Здоровая, 1 - Средняя, 2 - Нездоровая)",
    "previous_heart_problems": "Предыдущие проблемы с сердцем (1 - да, 0 - нет)",
    "medication_use": "Использование медикаментов (1 - да, 0 - нет)",
    "stress_level": "Уровень стресса (от 1 до 10)",
    "sedentary_hours_per_day": "Часы сидячего образа жизни в день",
    "bmi": "Индекс массы тела (ИМТ)",
    "triglycerides": "Уровень триглицеридов",
    "physical_activity_days_per_week": "Дни физической активности в неделю",
    "sleep_hours_per_day": "Часы сна в сутки",
    "blood_sugar": "Уровень сахара в крови",
    "ck-mb": "Уровень CK-MB",
    "troponin": "Уровень тропонина",
    "gender": "Пол (1 - мужской, 0 - женский)",
    "systolic_blood_pressure": "Систолическое кровяное давление",
    "diastolic_blood_pressure": "Диастолическое кровяное давление"
}

# Этот будет обрабатывать GET-запросы (когда пользователь просто открывает страницу)
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Отображает главную страницу с формой для загрузки."""
    # Просто возвращаем HTML-шаблон. 'request' обязателен для Jinja2.
    return templates.TemplateResponse("index.html", 
                                        {
                                            "request": request, 
                                            "features": model_features,
                                            "descriptions": FEATURE_DESCRIPTIONS
                                        }
                                    )

# Этот будет обрабатывать POST-запросы (когда пользователь отправляет файл через форму)
@app.post("/", response_class=HTMLResponse)
async def predict_from_form(request: Request, file: UploadFile = File(...)):
    """Принимает файл из формы, делает предсказание и возвращает ту же страницу с результатами."""
    if not file.filename.endswith('.csv'):
        # Вместо возврата JSON ошибки, можно вернуть страницу с сообщением об ошибке
        return templates.TemplateResponse("index.html", 
                                            {
                                                "request": request, 
                                                "error": "Неверный формат файла.", 
                                                "features": model_features,
                                                "descriptions": FEATURE_DESCRIPTIONS
                                            })
    
    try:
        # --- Логика предсказания (та же, что и раньше) ---
        contents = await file.read()
        buffer = io.StringIO(contents.decode('utf-8'))
        data_to_predict = pd.read_csv(buffer)

        if 'id' not in data_to_predict.columns:
            return templates.TemplateResponse("index.html", 
                                            {
                                                "request": request, 
                                                "error": "В CSV файле отсутствует колонка 'id'.", 
                                                "features": model_features,
                                                "descriptions": FEATURE_DESCRIPTIONS
                                            })
        
        ids = data_to_predict['id']

        predictions = model.predict(data_to_predict.drop('id', axis=1))
        result_dict = dict(zip(ids, predictions))
        results_json_string = json.dumps(result_dict, indent=4)

        results_df = pd.DataFrame(
            list(result_dict.items()), 
            columns=['id', 'prediction']
            )
        results_csv_string = results_df.to_csv(index=False)

        # --- Ключевое отличие: Возвращаем HTML-страницу с результатами ---
        # Мы передаем в шаблон словарь 'results', который будет использован для отрисовки таблицы
        return templates.TemplateResponse("index.html", 
                                            {
                                                "request": request, 
                                                "results": result_dict, # Для HTML-таблицы
                                                "results_json": results_json_string,
                                                "results_csv": results_csv_string,
                                                "features": model_features,
                                                "descriptions": FEATURE_DESCRIPTIONS # Для JavaScript-кнопки
                                            })

    except Exception as e:
        # В случае любой другой ошибки, также возвращаем страницу с сообщением
        return templates.TemplateResponse("index.html", 
                                            {
                                                "request": request, 
                                                "error": f"Произошла ошибка: {e}", 
                                                "features": model_features,
                                                "descriptions": FEATURE_DESCRIPTIONS
                                            })