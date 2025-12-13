# DRAGON Network (Distributed Routing and GPU Open Network)
> Note: Any references to `docs/features/*` are placeholders. Use `docs/INDEX.md` and `docs/guide/` for current documentation.

**Complete Implementation Guide**

**Status**: 🟡 Planning Phase  
**Date**: October 18, 2025  
**Priority**: High  
**Complexity**: Very High  

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Core Concept](#core-concept)
3. [System Architecture](#system-architecture)
4. [Technical Components](#technical-components)
5. [User Experience](#user-experience)
6. [Backend Infrastructure](#backend-infrastructure)
7. [Security & Privacy](#security-privacy)
8. [Implementation Phases](#implementation-phases)
9. [Implementation Checklist](#implementation-checklist)
10. [Technical Quick Reference](#technical-quick-reference)
11. [Testing & Deployment](#testing-deployment)
12. [Success Metrics](#success-metrics)

---

## Executive Summary

### What is DRAGON?

DRAGON is a crowd-sourced distributed training system that enables AI-OS users to donate their GPU/CPU time toward training a massive collaborative HRM-sMoE model. Users can participate with a simple one-click interface that automatically downloads training data batches, trains on locally allocated resources, and uploads results to a central aggregation server.

**Vision**: Transform AI-OS into a decentralized AI training network where thousands of volunteers collectively train state-of-the-art models.

### Key Value Propositions

1. **Democratic AI Training**: Anyone with a GPU can contribute to cutting-edge AI research
2. **Cost Efficiency**: Distribute training costs across community (200x cheaper than cloud)
3. **Accessibility**: One-click interface, no technical expertise required
4. **Community Building**: Gamification, leaderboards, and shared success
5. **Innovation**: Enable training of models too large for individual users

---

## Core Concept

### Federated Learning Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    DRAGON Aggregation Server                 │
│                                                              │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │ Model Registry │  │ Gradient Queue  │  │ Aggregator   │ │
│  │ (current model)│  │ (worker updates)│  │ (FedAvg/Adam)│ │
│  └────────────────┘  └─────────────────┘  └──────────────┘ │
│            │                  ▲                    │         │
└────────────┼──────────────────┼────────────────────┼─────────┘
             │                  │                    │
             ▼                  │                    ▼
   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
   │   Worker A       │  │   Worker B       │  │   Worker C       │
   │  (RTX 4090)      │  │  (RTX 3080)      │  │  (GTX 1080 Ti)   │
   │                  │  │                  │  │                  │
   │  1. Download     │  │  1. Download     │  │  1. Download     │
   │     model        │  │     model        │  │     model        │
   │  2. Fetch batch  │  │  2. Fetch batch  │  │  2. Fetch batch  │
   │  3. Train local  │  │  3. Train local  │  │  3. Train local  │
   │  4. Upload grads │  │  4. Upload grads │  │  4. Upload grads │
   └──────────────────┘  └──────────────────┘  └──────────────────┘
```

### How It Works

1. **Distributing Work**: Breaking training into small, manageable batches
2. **Volunteer Computing**: Users donate idle GPU/CPU time
3. **Gradient Aggregation**: Central server combines updates from all workers
4. **Simple UX**: One-button start/stop interface
5. **Resource Aware**: Respects user-defined GPU/CPU limits from Resources tab

---

## System Architecture

### Architecture: Centralized Aggregation Server (Recommended)

**Pros**:
- Simple to implement and debug
- Full control over aggregation algorithm
- Easy to monitor training progress
- Standard federated learning approach

**Cons**:
- Single point of failure
- Server bandwidth requirements
- Hosting costs

**Components**:

1. **DRAGON Server** (Python/FastAPI)
   - Model registry (current global model)
   - Batch distribution queue
   - Gradient aggregation service
   - Worker authentication and stats
   - Progress tracking

2. **DRAGON Client** (AI-OS GUI Tab)
   - Download model checkpoint
   - Request training batch
   - Local training loop
   - Gradient computation and upload
   - Automatic retry logic

3. **Communication Protocol** (REST API + WebSockets)
   - REST: Model downloads, batch requests, gradient uploads
   - WebSocket: Real-time status updates, heartbeat

---

## Technical Components

### 1. DRAGON GUI Tab

**Location**: `src/aios/gui/components/dragon_panel.py`

**UI Layout**:

```
┌────────────────────────────────────────────────────────────────┐
│                        DRAGON Network                          │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  Status: ●IDLE / ●TRAINING / ●UPLOADING                       │
│                                                                │
│  Global Model: HRM-sMoE-125M (v1.2.3)                         │
│  Your Contribution: 1,234 batches (12.3M tokens)              │
│  Network Stats: 523 active workers, 45.2B tokens processed    │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                Resource Allocation                        │ │
│  │                                                           │ │
│  │  Use settings from Resources tab: ☑                      │ │
│  │  Override GPU memory limit: [ 80% ]                      │ │
│  │  Max batch size: [ 4 ]                                   │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                Training Progress                          │ │
│  │                                                           │ │
│  │  Current batch: 15/100 (15%)                             │ │
│  │  [████████░░░░░░░░░░░░░░░░░░░░░]                         │ │
│  │                                                           │ │
│  │  Tokens processed: 15,234                                │ │
│  │  Loss: 2.456                                             │ │
│  │  Speed: 1,245 tok/s                                      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│       [  START TRAINING  ]        [  STOP  ]                  │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │                Activity Log                               │ │
│  │                                                           │ │
│  │  [12:34:56] Connected to DRAGON server                   │ │
│  │  [12:35:01] Downloaded model v1.2.3 (125M params)        │ │
│  │  [12:35:05] Fetched batch #1234 (1000 samples)           │ │
│  │  [12:35:45] Training complete (loss: 2.456)              │ │
│  │  [12:35:48] Uploaded gradients (2.3 MB)                  │ │
│  │  [12:35:50] Contribution recorded ✓                      │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 2. DRAGON Client Module

**Location**: `src/aios/dragon/client.py`

**Key Methods**:

```python
class DRAGONClient:
    def __init__(self, server_url: str, api_key: str):
        """Initialize connection to DRAGON server."""
        
    async def connect(self) -> bool:
        """Authenticate with server, verify connection."""
        
    async def download_model(self, version: str) -> Path:
        """Download global model checkpoint to cache."""
        
    async def fetch_batch(self) -> Dict:
        """Request next training batch from server."""
        
    async def train_batch(self, batch: Dict) -> Dict:
        """Train on batch, compute gradients."""
        
    async def upload_gradients(self, gradients: Dict, metadata: Dict):
        """Upload computed gradients to server."""
        
    async def start_training_loop(self):
        """Main training loop: fetch → train → upload."""
```

### 3. DRAGON Server (Backend)

**Technology Stack**:
- **FastAPI** - REST API framework
- **PostgreSQL** - Database for workers, batches, model versions
- **Redis** - Queue for batch distribution, gradient aggregation
- **S3/MinIO** - Object storage for model checkpoints
- **Docker** - Containerized deployment

**API Endpoints**:

```
POST   /api/v1/auth/register          # Register new worker
POST   /api/v1/auth/login             # Authenticate worker
GET    /api/v1/model/current          # Get current global model version
GET    /api/v1/model/download/{ver}   # Download model checkpoint
POST   /api/v1/batch/request          # Request next training batch
POST   /api/v1/gradients/upload       # Upload computed gradients
GET    /api/v1/stats/global           # Get network statistics
GET    /api/v1/worker/stats           # Get worker's contribution stats
WS     /api/v1/ws/status              # WebSocket for real-time updates
```

**Database Schema**:

```sql
-- Workers table
CREATE TABLE workers (
    id UUID PRIMARY KEY,
    api_key VARCHAR(64) UNIQUE NOT NULL,
    username VARCHAR(64),
    hardware_info JSONB,
    total_batches INT DEFAULT 0,
    total_tokens BIGINT DEFAULT 0,
    reputation_score FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT NOW(),
    last_seen TIMESTAMP
);

-- Model versions table
CREATE TABLE model_versions (
    version VARCHAR(16) PRIMARY KEY,
    checkpoint_url TEXT NOT NULL,
    num_parameters BIGINT,
    created_at TIMESTAMP DEFAULT NOW(),
    metrics JSONB
);

-- Training batches table
CREATE TABLE training_batches (
    batch_id UUID PRIMARY KEY,
    dataset_name VARCHAR(128),
    data_url TEXT NOT NULL,
    num_samples INT,
    status VARCHAR(32),  -- 'pending', 'assigned', 'completed', 'failed'
    assigned_to UUID REFERENCES workers(id),
    assigned_at TIMESTAMP,
    completed_at TIMESTAMP,
    result_hash VARCHAR(64)
);

-- Gradient updates table
CREATE TABLE gradient_updates (
    update_id UUID PRIMARY KEY,
    worker_id UUID REFERENCES workers(id),
    batch_id UUID REFERENCES training_batches(batch_id),
    model_version VARCHAR(16),
    gradient_url TEXT NOT NULL,
    loss FLOAT,
    num_samples INT,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    aggregated BOOLEAN DEFAULT FALSE
);
```

### 4. Aggregation Algorithm

**FedAvg (Federated Averaging)**:

```python
def aggregate_gradients(gradient_updates: List[GradientUpdate]) -> ModelUpdate:
    """
    Aggregate gradients from multiple workers using weighted averaging.
    
    Args:
        gradient_updates: List of gradient updates from workers
        
    Returns:
        Aggregated model update
    """
    # Calculate total samples across all workers
    total_samples = sum(update.num_samples for update in gradient_updates)
    
    # Initialize aggregated gradients dictionary
    aggregated_gradients = {}
    
    # For each parameter in the model
    for param_name in gradient_updates[0].gradients.keys():
        weighted_sum = 0
        
        # Weighted average based on number of samples
        for update in gradient_updates:
            weight = update.num_samples / total_samples
            weighted_sum += weight * update.gradients[param_name]
        
        aggregated_gradients[param_name] = weighted_sum
    
    return ModelUpdate(gradients=aggregated_gradients)
```

---

## User Experience

### User Journey: First-Time Contributor

1. **Discovery**:
   - User opens AI-OS GUI
   - Sees new "DRAGON" tab with "Contribute to AI Training"
   - Clicks to explore

2. **Registration** (One-time):
   - Click "Join DRAGON Network"
   - Enter optional username
   - System generates API key automatically
   - Connection test → Success ✓

3. **Training Session**:
   - Click "START TRAINING"
   - Progress bar shows batch download
   - Training begins (see real-time loss, speed)
   - Activity log shows each step
   - User can minimize or use other tabs

4. **Stopping**:
   - User clicks "STOP"
   - Current batch finishes
   - Gradients uploaded
   - Model unloaded from memory

5. **Viewing Contribution**:
   - "Your Contribution" section shows stats
   - Leaderboard shows rank
   - Badges for milestones (1K, 10K, 100K batches)

### Resource Management Integration

**Automatic Mode** (default):
- Reads GPU/CPU settings from Resources tab
- Respects memory limits
- Uses configured GPUs only

**Override Mode**:
- User can set DRAGON-specific limits
- Example: "Use 80% GPU for regular tasks, 60% for DRAGON"

---

## Backend Infrastructure

### Deployment Architecture

**Recommended Setup**:

1. **Application Server** (FastAPI + Gunicorn)
   - 4 CPU cores, 8GB RAM
   - Handles API requests
   - Load balancer for high availability

2. **Database** (PostgreSQL)
   - 2 CPU cores, 4GB RAM
   - Persistent storage for metadata
   - Read replicas for analytics

3. **Message Queue** (Redis)
   - 2 CPU cores, 4GB RAM
   - Persistent storage for work queue

4. **Object Storage** (S3/MinIO)
   - Store model checkpoints
   - Store gradient files
   - CDN for fast downloads

5. **Monitoring**:
   - Prometheus + Grafana for metrics
   - Alerting for failures
   - CloudWatch/similar for logs

**Estimated Costs** (AWS):
- Development: ~$50/month
- Production (100 workers): ~$200/month
- Production (1000 workers): ~$500/month

### Data Distribution Strategy

**Hybrid Approach** (Recommended):
- Pre-chunk common sizes (256, 512, 1024 samples)
- Generate custom sizes on-demand
- Cache in Redis for fast distribution
- Store in S3 for durability

---

## Security & Privacy

### Critical Security Measures

1. **No Code Execution on Workers**
   - Workers only receive model weights (safetensors format)
   - No pickle, eval(), or dynamic code
   - No eval() or pickle usage

2. **Gradient Validation**
   - Server checks gradient shapes match model architecture
   - Outlier detection (reject if > 3σ from mean)
   - Rate limiting to prevent poisoning attacks

3. **Authentication**
   - API keys generated client-side
   - Rate limiting per worker
   - Optional OAuth for verified contributors

4. **Byzantine Fault Tolerance**
   - Reputation scoring (workers build trust over time)
   - Krum aggregation (reject outliers)
   - Random batch verification (server trains same batch to validate)

5. **Data Privacy**
   - Training data is public domain (no PII)
   - No user data uploaded
   - Differential privacy (future enhancement)

### Attack Vectors & Mitigations

| Attack | Mitigation |
|--------|------------|
| **Gradient Poisoning** | Outlier detection, Krum aggregation, reputation scores |
| **Model Extraction** | Rate limiting, partial model access (MoE experts only) |
| **Sybil Attack** | CAPTCHA on registration, proof-of-work, rate limits |
| **DDoS** | Cloudflare, rate limiting, worker throttling |
| **Data Poisoning** | Curated public datasets only, no user-uploaded data |

---

## Implementation Phases

### Phase 1: MVP (1-2 months)

**Goal**: Proof of concept with basic functionality

**Features**:
- Simple DRAGON tab in GUI
- FastAPI server with basic endpoints
- Single global model (no versioning)
- FedAvg aggregation
- Manual start/stop only
- Fixed batch size (512 samples)
- 10-50 test workers

**Deliverables**:
- `src/aios/dragon/client.py`
- `src/aios/gui/components/dragon_panel.py`
- `backend/dragon_server/` (FastAPI app)
- Docker compose for local testing
- Documentation: setup guide, API reference

### Phase 2: Production Ready (2-3 months)

**Features**:
- Model versioning and rollback
- Dynamic batch sizing
- Automatic retry logic
- Worker reputation system
- Real-time monitoring dashboard
- Progress persistence (resume after restart)
- 100-500 workers

**Enhancements**:
- WebSocket for live updates
- Gradient compression (reduce upload size)
- Checkpoint caching (avoid re-downloads)
- Advanced aggregation (FedAdam)

### Phase 3: Scale & Optimize (3-6 months)

**Features**:
- Auto-scaling infrastructure
- 1000+ concurrent workers
- Expert-level distribution (different workers train different MoE experts)
- Multi-model support (users choose which model to contribute to)
- Contribution leaderboard and gamification
- Mobile app for cross-platform training

---

## Implementation Checklist

### Pre-Implementation Setup

**Documentation**:
- [x] Complete architectural plan
- [x] Technical quick reference
- [ ] API specification (OpenAPI/Swagger)
- [ ] Database schema SQL scripts
- [ ] Frontend mockups (wireframes)

**Research**:
- [ ] Review FedAvg paper (McMahan et al., 2017)
- [ ] Review FedAdam paper (Reddi et al., 2020)
- [ ] Benchmark gradient compression methods
- [ ] Survey existing platforms (Flower, PySyft)

**Environment Setup**:
- [ ] Set up development server (local Docker)
- [ ] Configure PostgreSQL database
- [ ] Configure Redis message queue
- [ ] Set up S3/MinIO for object storage
- [ ] Configure monitoring (Prometheus + Grafana)

### Backend Development

**Core Infrastructure**:
- [ ] Initialize FastAPI project structure
- [ ] Create database migration system (Alembic)
- [ ] Implement database tables
- [ ] Add indexes for performance
- [ ] Seed initial data

**API Endpoints**:
- [ ] `POST /api/v1/auth/register` - Worker registration
- [ ] `POST /api/v1/auth/login` - Authentication
- [ ] `GET /api/v1/model/current` - Get latest model
- [ ] `GET /api/v1/model/download/{ver}` - Download checkpoint
- [ ] `POST /api/v1/batch/request` - Request training batch
- [ ] `POST /api/v1/gradients/upload` - Upload gradients
- [ ] `GET /api/v1/stats/global` - Network statistics
- [ ] `GET /api/v1/worker/stats` - Worker stats

**Aggregation System**:
- [ ] Implement FedAvg algorithm
- [ ] Aggregation scheduler (every 60s or 10 updates)
- [ ] Model version management
- [ ] Gradient queue management

**Testing**:
- [ ] Unit tests for API endpoints
- [ ] Integration tests for aggregation
- [ ] Load tests (100 concurrent requests)

### Client Development

**Core Module**:
- [ ] Create `src/aios/dragon/` package
- [ ] Implement `DRAGONClient` class
- [ ] `connect()` - authenticate with server
- [ ] `download_model()` - fetch checkpoint
- [ ] `fetch_batch()` - request training data
- [ ] `train_batch()` - local training loop
- [ ] `upload_gradients()` - send results
- [ ] `start_training_loop()` - main loop

**Training Integration**:
- [ ] Adapt `train_epoch.py` for single-batch training
- [ ] Gradient extraction and serialization
- [ ] Checkpoint management

**Resource Integration**:
- [ ] Read GPU limits from Resources panel
- [ ] Auto-detect batch size
- [ ] Error handling & retry logic

**Testing**:
- [ ] Unit tests for `DRAGONClient`
- [ ] Mock server for integration tests
- [ ] End-to-end test

### GUI Development

**DRAGON Panel**:
- [ ] Create `src/aios/gui/components/dragon_panel.py`
- [ ] Implement `DRAGONPanel(ttk.Frame)`
- [ ] Status indicator widget
- [ ] Progress section
- [ ] Stats section
- [ ] Controls section

**Integration**:
- [ ] Add DRAGON tab to main app
- [ ] Initialize DRAGON panel
- [ ] Event handlers (start, stop, update)

**Testing**:
- [ ] Manual testing (visual inspection)
- [ ] Start/stop functionality
- [ ] Progress updates accuracy

### Integration Testing

- [ ] Test with 1 worker (local)
- [ ] Test with 2 workers (different batches)
- [ ] Test with 5 workers (concurrent)
- [ ] Worker disconnect during training
- [ ] Server restart during aggregation
- [ ] Network interruption handling

### Deployment Preparation

- [ ] Create Docker configuration
- [ ] Set up reverse proxy (nginx)
- [ ] Configure SSL certificates
- [ ] Set up database backups
- [ ] Configure logging
- [ ] Set up error tracking

---

## Technical Quick Reference

### Core Files

| Component | File | Purpose |
|-----------|------|---------|
| GUI Tab | `src/aios/gui/components/dragon_panel.py` | User interface |
| Client | `src/aios/dragon/client.py` | Server communication |
| Server | `backend/dragon_server/main.py` | API endpoints |
| Database | `backend/dragon_server/models.py` | Schema definitions |
| Aggregation | `backend/dragon_server/aggregator.py` | FedAvg implementation |

### Data Flow

```
1. Worker clicks "START TRAINING"
2. Client connects to server (auth)
3. Download current global model checkpoint
4. Request training batch from work queue
5. Train locally (compute gradients)
6. Upload gradients to server
7. Server aggregates using FedAvg
8. Repeat steps 4-7 until user clicks "STOP"
```

### API Request/Response Examples

**Request Batch**:
```json
POST /api/v1/batch/request
{
  "worker_id": "uuid",
  "hardware": {
    "gpu": "RTX 4090",
    "vram_gb": 24,
    "batch_size": 8
  }
}

Response:
{
  "batch_id": "uuid",
  "model_version": "v1.2.3",
  "data_url": "https://s3.../batch.jsonl.gz",
  "num_samples": 1000
}
```

**Upload Gradients**:
```json
POST /api/v1/gradients/upload
{
  "worker_id": "uuid",
  "batch_id": "uuid",
  "model_version": "v1.2.3",
  "gradients_url": "https://s3.../grads.safetensors",
  "loss": 2.456,
  "num_samples": 1000
}
```

### Configuration

**Client Config** (`config/dragon.yaml`):
```yaml
server:
  url: "https://dragon.aios.ai"
  api_key: "generated_on_first_run"

training:
  batch_size: auto
  max_batches_per_session: 100
  retry_attempts: 3
  timeout_seconds: 300

uploads:
  compress_gradients: true
  compression_ratio: 0.1
```

**Server Config** (`.env`):
```env
DATABASE_URL=postgresql://user:pass@localhost/dragon
REDIS_URL=redis://localhost:6379
S3_BUCKET=dragon-checkpoints
AGGREGATION_INTERVAL=60
MIN_GRADIENTS_FOR_UPDATE=10
```

---

## Testing & Deployment

### Testing Strategy

**Unit Tests**:
- Client methods (`test_dragon_client.py`)
- Aggregation correctness (`test_aggregation.py`)
- Security validation (`test_security.py`)

**Integration Tests**:
- 2-worker local test
- Server aggregation
- Fault tolerance

**Load Tests**:
- 100 concurrent workers
- 1000 requests/second
- 10 GB/hour gradient uploads

### Monitoring

**Metrics to Track**:
- Active workers (gauge)
- Batches trained per hour (counter)
- Average loss per model version (gauge)
- Gradient upload success rate (%)
- Server response time (p50, p95, p99)

**Alerts**:
- Worker count drops by >50%
- Average loss increases by >10%
- Server error rate >1%
- Gradient upload failures >5%

### Deployment

**Development** (Docker Compose):
```bash
cd backend/dragon_server
docker-compose up -d
```

**Production** (Kubernetes):
```bash
kubectl apply -f k8s/dragon-server.yaml
```

---

## Success Metrics

### Phase 1 MVP Success

- [ ] 10+ workers successfully training
- [ ] Model converging (loss decreasing)
- [ ] Zero critical bugs
- [ ] Average uptime >95%

### Phase 2 Production Success

- [ ] 100+ active workers
- [ ] 1M+ tokens processed per day
- [ ] Worker retention >30%
- [ ] Infrastructure cost <$200/month

### Long-term Impact Metrics

- **Community Growth**: 1000+ active contributors
- **Training Efficiency**: 10B+ tokens/month
- **Cost Savings**: $10K+ equivalent compute donated
- **Model Quality**: Competitive with commercial models
- **Innovation**: 5+ research papers using DRAGON

---

## FAQ

**Q: How long does training one batch take?**  
A: ~2 minutes on RTX 3090 (1000 samples, batch_size=4)

**Q: How much bandwidth is needed?**  
A: ~660 MB/hour (model cached, 12 batches/hour × 55 MB)

**Q: What if my internet disconnects?**  
A: Retry logic auto-reconnects. Current batch is lost, but resume from next batch.

**Q: Can I pause and resume later?**  
A: Yes! Click STOP, your progress is saved. Click START to continue.

**Q: Do I get credit for my contribution?**  
A: Yes! Leaderboard tracks your batches and reputation score.

**Q: Is my data private?**  
A: Yes. Only gradients are uploaded, no personal data leaves your machine.

**Q: What hardware do I need?**  
A: Any NVIDIA GPU with 4GB+ VRAM. CPU-only mode supported but slower.

---

## Next Steps

### Immediate Actions (Week 1)

1. **Stakeholder Review**: Present this plan to AI-OS maintainers
2. **Environment Setup**: Configure development server (Docker)
3. **Team Assembly**: Assign roles (backend, client, GUI, testing)

### Short-term Goals (Month 1)

1. **Backend MVP**: FastAPI server with core endpoints
2. **Client MVP**: Basic training loop working
3. **GUI MVP**: Simple tab with start/stop buttons
4. **Testing**: End-to-end test with 2 workers

### Medium-term Goals (Months 2-3)

1. **Alpha Testing**: 20-50 internal users
2. **Production Hardening**: Error handling, monitoring, security
3. **Documentation**: User guide, API docs, troubleshooting
4. **Beta Launch**: Public announcement, community onboarding

---

## References

### Papers
- McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data" (FedAvg)
- Reddi et al., "Adaptive Federated Optimization" (FedAdam)
- Blanchard et al., "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent" (Krum)

### Existing Platforms
- **Flower**: Federated learning framework
- **PySyft**: Privacy-preserving ML
- **BOINC**: Volunteer computing platform
- **Folding@home**: Distributed protein folding

### AI-OS Integration Points
- Streaming Datasets: (placeholder) `docs/DATASET_STREAM_QUEUE.md`
- Checkpoint System: (placeholder) `docs/features/AUTOMATIC_CHECKPOINT_SAVING.md`
- Resource Management: (placeholder) `docs/fixes/RESOURCES_TAB_SETTINGS_PERSISTENCE_FIX.md`
- HRM-sMoE: (placeholder) `docs/HRM_MOE_SUMMARY.md`

Note: The above items are planning placeholders for docs that are not yet created or published. For current docs, start at `docs/INDEX.md` and explore `docs/guide/`.

---

**Document Version**: 1.0  
**Last Updated**: October 18, 2025  
**Status**: 🟢 Ready for Implementation  

---

## Appendix: Code Examples

### Example: Client Training Loop

```python
async def training_loop(self):
    """Main DRAGON training loop."""
    while not self.stop_flag:
        try:
            # 1. Fetch batch
            batch = await self.fetch_batch()
            self.update_status("TRAINING")
            
            # 2. Train locally
            gradients, loss = await self.train_batch(batch)
            
            # 3. Upload results
            self.update_status("UPLOADING")
            await self.upload_gradients(gradients, {
                "batch_id": batch["batch_id"],
                "loss": loss,
                "num_samples": len(batch["data"])
            })
            
            # 4. Update stats
            self.total_batches += 1
            self.update_progress()
            
        except Exception as e:
            self.log_error(f"Training failed: {e}")
            await asyncio.sleep(30)  # Retry after 30s
    
    self.update_status("IDLE")
```

### Example: Server Aggregation

```python
@app.post("/api/v1/aggregate")
async def trigger_aggregation():
    """Aggregate pending gradients and update global model."""
    
    # Fetch pending gradients
    updates = await db.fetch_pending_gradients(limit=100)
    
    if len(updates) < MIN_GRADIENTS_FOR_UPDATE:
        return {"status": "waiting", "pending": len(updates)}
    
    # Aggregate using FedAvg
    aggregated = aggregate_gradients(updates)
    
    # Update global model
    current_model = await load_model()
    current_model.apply_gradients(aggregated)
    
    # Save new version
    new_version = increment_version()
    checkpoint_url = await save_checkpoint(current_model, new_version)
    
    # Mark gradients as aggregated
    await db.mark_aggregated(updates)
    
    return {
        "status": "success",
        "version": new_version,
        "num_updates": len(updates)
    }
```

---

**End of Document**
