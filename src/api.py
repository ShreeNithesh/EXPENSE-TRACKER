"""API for Expense Category Predictor using FastAPI."""

from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel

from src.prediction import predict_category, load_artifacts


app = FastAPI(title="Expense Category Predictor")


class PredictRequest(BaseModel):
    merchant: str
    amount: float


class PredictResponse(BaseModel):
    category: str
    confidence: float

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root(docs: Optional[bool] = False):
    """Root endpoint: returns a small message or redirects to docs if requested."""
    # If the caller wants docs via query param ?docs=1, redirect there
    if docs:
        return RedirectResponse(url='/docs')
    return {"service": "Expense Category Predictor", "docs": "/docs", "health": "/health"}


@app.get("/favicon.ico")
async def favicon():
    # Return no content for favicon requests (prevents 404 noise)
    return Response(status_code=204)

@app.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        category, confidence = predict_category(req.merchant, req.amount)
        return PredictResponse(category=category, confidence=confidence)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload")
async def reload_models():
    try:
        # call load_artifacts to force reload (it reads disk files each time)
        load_artifacts()
        return {"status": "reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
