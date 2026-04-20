import os
import sys
from src.exception import CustomException
from src.pipline.predict_pipline import PredictPipeline, CustomData
 
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
 
app = FastAPI()
 
templates = Jinja2Templates(directory="templates")
 
 
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")
 
 
@app.post("/predict")
def predict(data: CustomData):
    try:
        df = data.to_dataframe()
        print(df)
        pipeline = PredictPipeline()
        preds = pipeline.predict(df)
        return {"preds": preds[0]}
    except CustomException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")