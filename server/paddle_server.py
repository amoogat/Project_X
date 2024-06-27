from fastapi import FastAPI
from pydantic import BaseModel
from paddleocr import PaddleOCR
from typing import List
import logging
import uvicorn

app = FastAPI()

# Initialize the PaddleOCR model with GPU support
ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

class OCRRequest(BaseModel):
    image_urls: List[str]

@app.post("/ocr/")
async def ocr_image(request: OCRRequest):
    results = []
    fixed_dim = 2000
    scale_factor = 1000

    for url in request.image_urls:
        if not url:
            results.append({"url": url, "text": None})
            continue
        try:
            ocr_result = ocr.ocr(url, cls=True)

            res_string = ''
            for idx in range(len(ocr_result)):
                res = ocr_result[idx]
                for line in res:
                    text = line[1][0]
                    bounding_box = line[0]
                    if any(isinstance(coord, list) for coord in bounding_box):
                        bounding_box = [item for sublist in bounding_box for item in sublist]
                    normalized_box = [((coord / fixed_dim) * scale_factor) for coord in bounding_box]
                    box_pairs = zip(*[iter(normalized_box)]*2)
                    box_str = ','.join([f"({x:.0f},{y:.0f})" for x, y in box_pairs])
                    res_string += f"Box:{box_str} Text:{text}"
                    logging.info(f"Box:{box_str} Text:{text}")

            results.append({"url": url, "text": res_string})
        except Exception as e:
            logging.error(f"An error occurred while processing {url}: {str(e)}")
            results.append({"url": url, "text": None})

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
