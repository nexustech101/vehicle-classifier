# Quick Deployment Guide

## 🚀 Start API in 30 Seconds

### Option 1: FastAPI (Development)

```bash
# Install dependencies
pip install -r requirements.txt

# Start server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# Navigate to: http://localhost:8000/docs
```

### Option 2: Docker Compose (Production)

```bash
# Start all services
docker-compose up -d

# Verify
docker-compose ps

# View logs
docker-compose logs -f app

# Stop
docker-compose down
```

---

## 📋 API Endpoints

```bash
# Health check
curl http://localhost:8000/health

# Get metadata
curl http://localhost:8000/api/models/metadata

# Classify single image
curl -X POST -F "file=@vehicle.jpg" http://localhost:8000/api/vehicle/classify

# Batch processing
curl -X POST -F "files=@car1.jpg" -F "files=@car2.jpg" \
  http://localhost:8000/api/vehicle/classify-batch

# Generate HTML report
curl -X POST -F "file=@vehicle.jpg" -F "format=html" \
  http://localhost:8000/api/vehicle/report > report.html

# Interactive docs
open http://localhost:8000/docs
```

---

## 📊 View Logs

```bash
# API logs (real-time)
tail -f logs/api.log

# Training logs
tail -f logs/training.log

# Parse JSON (pretty-print)
cat logs/api.log | jq '.'
```

---

## 🔧 Environment Variables

**Development:**
```bash
export REDIS_HOST=localhost
export REDIS_PORT=6379
export LOG_LEVEL=INFO
```

**Docker:**
Set in `docker-compose.yml`:
```yaml
environment:
  - REDIS_HOST=redis
  - REDIS_PORT=6379
  - LOG_LEVEL=INFO
```

---

## 📦 Project Structure

```
├── src/
│   ├── api/
│   │   ├── app.py              # FastAPI application
│   │   ├── service.py          # API service layer
│   │   ├── cache.py            # Redis caching
│   │   └── logging_config.py   # Structured logging
│   ├── models/
│   │   └── classifiers.py      # ML models (transfer learning)
│   ├── training/
│   │   └── train.py            # Training pipeline
│   └── preprocessing/
│       ├── processor.py        # Image processing
│       └── utils.py            # Utilities
├── docker-compose.yml          # Container orchestration
├── Dockerfile                  # FastAPI image
├── requirements.txt            # Dependencies
└── README.md                   # Full documentation
```

---

## ✨ Key Features

- 🎯 **9 Vehicle Classifiers** - Make, Type, Color, Condition, etc.
- ⚡ **FastAPI** - Async REST API with auto-generated docs
- 💾 **Redis** - Distributed caching with regional analytics
- 🐳 **Docker** - Production-ready containerization
- 📊 **Structured Logging** - JSON logs for observability
- 🔄 **Transfer Learning** - EfficientNet backbone models
- 📈 **Batch Processing** - Multi-image classification
- 🎨 **Report Generation** - JSON/HTML professional reports

---

## 🐛 Troubleshooting

**API won't start?**
```bash
# Check port 8000 is free
lsof -i :8000

# Check dependencies
python -c "import fastapi; print('✓ FastAPI installed')"
```

**Redis connection error?**
```bash
# Redis is optional - API works without it (caching disabled)
# To fix Redis connection in Docker:
docker-compose down -v
docker-compose up -d
```

**Model loading fails?**
```bash
# Verify checkpoints directory
ls -la checkpoints/

# Check logs
tail -f logs/api.log
```

---

## 📞 Support

- **API Docs**: http://localhost:8000/docs
- **Logs**: See `logs/` directory
- **README**: See `README.md` for detailed documentation
- **Implementation**: See `INFRASTRUCTURE_UPDATE.md` for technical details
