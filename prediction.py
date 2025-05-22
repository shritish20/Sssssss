from fastapi import APIRouter
from utils.xgb_model import run_xgb_prediction

router = APIRouter()

@router.get("/predict-volatility")
def predict_volatility_api():
    try:
        prediction = run_xgb_prediction()
        return {"status": "success", "prediction": prediction}
    except Exception as e:
        return {"status": "error", "message": str(e)}