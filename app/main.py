from fastapi import FastAPI
app= FastAPI(title="Semantic Caching Gateway")


@app.get("/health")
async def health():
    return {"status":"healthy"}

