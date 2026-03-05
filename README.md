# Business Judgment System using Fine-Tuned LLM with RAG

A decision-support system that provides explainable, case-based business failure analysis using fine-tuned Qwen2.5-1.5B with Retrieval-Augmented Generation (RAG).

## 🎯 Overview

Traditional ML models predict failure probabilities but fail to explain *why* failure occurs or *what to do next*. This system combines fine-tuning and RAG to deliver actionable judgments grounded in historical precedents.

### Key Features

- **Explainable Analysis**: Structured reasoning instead of black-box predictions
- **Historical Grounding**: RAG retrieves similar cases to reduce hallucinations
- **Actionable Insights**: Clear guidance on what will/won't work
- **Local Inference**: Runs entirely on local hardware (M1/M2 Mac or GPU)
- **Efficient Training**: QLoRA enables fine-tuning on consumer hardware

## 🏗️ Architecture

```
┌─────────────────┐
│  User Scenario  │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  RAG Retrieval (FAISS)          │
│  • Semantic search              │
│  • Top-3 similar cases          │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Fine-Tuned Qwen2.5-1.5B        │
│  • QLoRA adapters               │
│  • Reasoning-trained            │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  Structured Judgment            │
│  • Core dynamics                │
│  • What could work              │
│  • What won't work              │
│  • Final judgment               │
│  • Reference cases              │
└─────────────────────────────────┘
```

## 📊 Data Pipeline

### Three Datasets, One Format

All heterogeneous data sources are transformed into reasoning narratives:

**Dataset 1: Startup Failures** (Qualitative)
- Raw indicators → Natural language narratives
- No raw features exposed to model
- Focus on failure patterns and dynamics

**Dataset 2: US Bankruptcy** (Financial Metrics)
- Financial ratios → Conceptual interpretations
- Labels removed immediately
- Narrative-based financial stress analysis

**Dataset 3: Long-Horizon Bankruptcy** (Temporal)
- Multi-year trajectories → Trend narratives
- Decay signals → Severity judgments
- Temporal reasoning patterns

### Unified Training Format

```
INPUT: Narrative scenario (no raw features)
OUTPUT: 
  - Core failure dynamics
  - What could work
  - What will not work
  - Final judgment (RECOVERABLE/CRITICAL/TERMINAL)
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.10+
# For training: CUDA GPU (RunPod A100 recommended)
# For inference: M1/M2 Mac or CUDA GPU
```

### Installation

```bash
# Clone repository
git clone <repository-url>
cd qwen-business-judgment

# Install dependencies
pip install torch transformers peft accelerate bitsandbytes
pip install sentence-transformers faiss-cpu gradio
pip install datasets tqdm pandas numpy
```

### 1. Prepare Data

```bash
cd run_pod
python prepare_data.py
```

**Output:**
- `data/processed/training_data.jsonl` (~78K samples)
- `data/processed/rag_corpus.jsonl` (~78K documents)

### 2. Fine-Tune Model

**On RunPod A100 (Recommended):**

```bash
# Upload project to /workspace/qwen-business-judgment
python run_pod/train.py
```

**Training Configuration:**
- Base Model: Qwen/Qwen2.5-1.5B
- Method: QLoRA (4-bit quantization)
- LoRA rank: 64, alpha: 16
- Epochs: 3
- Batch size: 8 (gradient accumulation: 2)
- Learning rate: 2e-4
- Time: ~2-3 hours on A100
- Cost: ~$4-6 on RunPod

**Output:**
- `models/qwen-business-judgment/` (LoRA adapters)

### 3. Build RAG Index

```bash
cd scripts
python build_rag_index.py
```

**Output:**
- `rag_index/faiss_index.bin` (FAISS index)
- `rag_index/documents.pkl` (document store)
- `rag_index/embeddings.npy` (embeddings)

### 4. Run Inference

```bash
cd scripts
python inference.py
```

**Launches Gradio UI at:** `http://localhost:7860`

## 💻 Usage

### Web Interface

1. Enter business scenario in text box
2. Toggle RAG retrieval (recommended: ON)
3. Adjust temperature (0.1-1.0, default: 0.7)
4. Set max tokens (256-2048, default: 1024)
5. Click "Submit"

### Example Input

```
A fintech startup that raised $50M but failed to achieve 
product-market fit after 3 years. The company faced intense 
competition from established banks and struggled with customer 
acquisition costs that exceeded lifetime value. After burning 
through capital, they were forced to shut down.
```

### Example Output

```
ANALYSIS:
The core failure mechanism centers on poor product-market fit 
compounded by unsustainable unit economics. Competing against 
entrenched players with asymmetric resources created an 
insurmountable barrier...

WHAT COULD WORK:
• Strategic retreat to defensible niche where incumbents have 
  structural disadvantages
• Complete business model redesign around proven monetization vector

WHAT WILL NOT WORK:
• Direct feature competition with incumbents: they will always 
  match and out-spend
• Growth-first strategy: scaling unprofitable unit economics 
  accelerates failure

FINAL JUDGMENT:
CRITICAL CONDITION: Company faces existential crisis requiring 
immediate, radical intervention.

---
RETRIEVED REFERENCE CASES:
Case 1 (relevance: 0.892): A payments company operated for 4 years...
Case 2 (relevance: 0.874): A lending platform raised $40M...
Case 3 (relevance: 0.861): A neobank struggled with CAC...
```

## 🔧 Technical Details

### Model Architecture

- **Base Model**: Qwen2.5-1.5B (1.5B parameters)
- **Fine-Tuning**: QLoRA with 4-bit quantization
- **LoRA Config**:
  - Rank: 64
  - Alpha: 16
  - Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  - Dropout: 0.1
- **Trainable Parameters**: ~50M (3.3% of base model)

### RAG System

- **Embedding Model**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS (IndexFlatIP with L2 normalization)
- **Retrieval**: Cosine similarity, top-3 results
- **Index Size**: ~78K documents, 384-dimensional embeddings

### Inference Configuration

- **Device**: MPS (M1/M2 Mac) or CUDA
- **Precision**: FP16 (MPS) or FP32 (CPU)
- **Context Window**: 2048 tokens
- **Generation**: Temperature sampling (default: 0.7)
- **Latency**: 10-20 seconds on M1 Pro

## 📁 Project Structure

```
qwen-business-judgment/
├── data/
│   ├── raw/                          # Original datasets
│   │   ├── dataset_paper/            # Startup failures CSVs
│   │   └── american_bankruptcy_dataset.csv
│   └── processed/
│       ├── training_data.jsonl       # Formatted training data
│       └── rag_corpus.jsonl          # RAG documents
├── models/
│   └── qwen-business-judgment/       # Fine-tuned LoRA adapters
│       ├── adapter_model.safetensors
│       ├── adapter_config.json
│       └── training_config.json
├── rag_index/
│   ├── faiss_index.bin               # FAISS vector index
│   ├── documents.pkl                 # Document metadata
│   └── embeddings.npy                # Embedding vectors
├── run_pod/
│   ├── prepare_data.py               # Data processing pipeline
│   └── train.py                      # QLoRA training script
├── scripts/
│   ├── build_rag_index.py            # RAG index builder
│   └── inference.py                  # Inference + Gradio UI
└── README.md
```

## 🎓 Why This Approach Works

### Fine-Tuning + RAG Synergy

| Component | Purpose | Benefit |
|-----------|---------|---------|
| **Fine-Tuning** | Teaches reasoning structure | Consistent judgment format |
| **RAG** | Provides specific examples | Grounded, defensible outputs |
| **Combined** | Pattern recognition + evidence | Explainable decisions |

### Traditional ML vs This System

| Traditional ML | This System |
|----------------|-------------|
| Probability score (0.73) | Structured judgment |
| No explanation | Detailed reasoning |
| Black box | Transparent with references |
| Not actionable | Clear next steps |

## 📈 Results

### Qualitative Improvements

✅ Clear references to relevant historical cases  
✅ More decisive judgments (RECOVERABLE/CRITICAL/TERMINAL)  
✅ Better separation of viable vs futile actions  
✅ Reduced hallucinations through RAG grounding  

### Performance Metrics

- **Training Samples**: 78,000+
- **RAG Documents**: 78,000+
- **Inference Latency**: 10-20s (M1 Pro)
- **Model Size**: 3GB (quantized with adapters)

## ⚠️ Limitations

### Where It Breaks

- **Novel Scenarios**: No historical analogs → weaker judgments
- **Quantitative Precision**: Not designed for exact financial calculations
- **Real-Time Data**: No access to current market conditions
- **Domain Boundaries**: Trained on startups/bankruptcies only

### Not Suitable For

❌ High-frequency trading decisions  
❌ Regulatory compliance automation  
❌ Legal liability determinations  
❌ Scenarios requiring real-time data  

## 🔮 Future Improvements

### Planned Enhancements

1. **Hybrid Retrieval**
   - Combine semantic + symbolic filters
   - Industry-specific retrieval
   - Temporal relevance weighting

2. **Enhanced Reasoning**
   - Tool-assisted financial calculations
   - Multi-step strategic planning
   - Scenario simulation capabilities

3. **Expanded Coverage**
   - Additional industries (healthcare, manufacturing)
   - International bankruptcy data
   - M&A failure analysis

4. **Production Features**
   - API endpoint deployment
   - Batch processing mode
   - Confidence calibration
   - Human-in-the-loop feedback

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `FileNotFoundError: RAG index not found`
```bash
# Solution: Build RAG index first
cd scripts
python build_rag_index.py
```

**Issue**: `CUDA out of memory` during training
```bash
# Solution: Reduce batch size in train.py
per_device_train_batch_size: int = 4  # Instead of 8
gradient_accumulation_steps: int = 4  # Instead of 2
```

**Issue**: Slow inference on Mac
```bash
# Solution: Reduce max_new_tokens
max_new_tokens=512  # Instead of 1024
```

**Issue**: Model outputs generic responses
```bash
# Solution: Enable RAG and lower temperature
use_rag=True
temperature=0.5  # Instead of 0.7
```

## 📚 Citation

If you use this system in your research, please cite:

```bibtex
@software{business_judgment_system,
  title={Business Judgment System using Fine-Tuned LLM with RAG},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/qwen-business-judgment}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Open a Pull Request

