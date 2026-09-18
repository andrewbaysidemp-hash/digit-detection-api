---
title: Digit Detection API
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Reads handwritten/printed digits from images (OpenCV + MNIST CNN, CPU)
---

# Digit Detection API

Upload an image of handwritten or printed digits and get the number back.

- Web page: `/`
- API docs: `/docs`
- JSON: `POST /api/v1/detect` with multipart field `image`
- Annotated PNG: `POST /api/v1/detect/annotated`
- Health: `/healthz`

```bash
curl -X POST "https://USER-SPACE.hf.space/api/v1/detect" -F "image=@digits.jpg"
```

Source and training code: see the project repository. Runs on CPU with ONNX
Runtime; no data leaves the container.
