from fastapi import FastAPI

app = FastAPI(title="DJ AI Assistant")

@app.get("/")
def read_root():
    return {"message": "DJ AI Assistant backend running"}
