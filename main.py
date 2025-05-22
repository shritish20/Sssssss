from fastapi import FastAPI
from routers import volguard, prediction, strategy

app = FastAPI(title="VolGuard Pro API")

app.include_router(volguard.router, prefix="/api")
app.include_router(prediction.router, prefix="/api")
app.include_router(strategy.router, prefix="/api")

@app.get("/")
def root():
    return {"message": "VolGuard Pro API is live"}