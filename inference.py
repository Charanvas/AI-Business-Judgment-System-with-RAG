"""
Local Inference with RAG - FIXED VERSION
Run on: MacBook M1 Pro
"""

import torch
import faiss
import pickle
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import gradio as gr
import os

class BusinessJudgmentSystem:
    """Complete inference system with RAG"""
    
    def __init__(
        self,
        adapter_path: str = "../models/qwen-business-judgment",
        rag_index_path: str = "../rag_index",
        device: str = "mps"  # "mps" for M1, "cpu" for Intel Mac
    ):
        self.device = device
        print(f"🖥️  Using device: {self.device}")
        
        # IMPORTANT: Base model name
        base_model_name = "Qwen/Qwen2.5-1.5B"
        
        # Convert to absolute path using Path
        adapter_path = str(Path(adapter_path).resolve())
        print(f"📁 Adapter path: {adapter_path}")
        
        # Load tokenizer from BASE MODEL (not adapter)
        print(f"📥 Loading tokenizer from base model: {base_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=True
        )
        print("✅ Tokenizer loaded")
        
        # Load base model
        print(f"📥 Loading base model: {base_model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16 if device == "mps" else torch.float32,
            device_map=device,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        print("✅ Base model loaded")
        
        # Load LoRA adapters
        print(f"📥 Loading LoRA adapters from: {adapter_path}")
        self.model = PeftModel.from_pretrained(
            self.model, 
            adapter_path,
            is_trainable=False,
            local_files_only=True
        )
        self.model.eval()
        
        print("✅ Model with adapters loaded successfully!")
        
        # Load RAG
        print("📚 Loading RAG index...")
        self.load_rag(rag_index_path)
        
    def load_rag(self, rag_index_path: str):
        """Load FAISS index and documents"""
        rag_path = Path(rag_index_path)
        
        index_file = rag_path / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"RAG index not found at {index_file}. Run build_rag_index.py first!")
        
        self.faiss_index = faiss.read_index(str(index_file))
        
        with open(rag_path / "documents.pkl", 'rb') as f:
            self.documents = pickle.load(f)
        
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        print(f"✅ RAG loaded: {len(self.documents)} documents")
        
    def retrieve_context(self, query: str, k: int = 3):
        """Retrieve relevant documents"""
        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True
        ).astype('float32')
        
        faiss.normalize_L2(query_embedding)
        distances, indices = self.faiss_index.search(query_embedding, k)
        
        retrieved_docs = []
        for idx, score in zip(indices[0], distances[0]):
            doc = self.documents[idx].copy()
            doc['score'] = float(score)
            retrieved_docs.append(doc)
        
        return retrieved_docs
    
    def generate_response(
        self,
        scenario: str,
        use_rag: bool = True,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ):
        """Generate judgment"""
        
        # Retrieve context if RAG enabled
        context = ""
        retrieved_docs = []
        
        if use_rag:
            print("🔍 Retrieving similar cases...")
            retrieved_docs = self.retrieve_context(scenario, k=3)
            context_parts = []
            for i, doc in enumerate(retrieved_docs):
                context_parts.append(f"Reference case {i+1}: {doc['text'][:200]}...")
            context = "\n\n".join(context_parts)
        
        # Build prompt
        system_prompt = """You are an expert business analyst specializing in failure analysis and strategic judgment. Your role is to provide clear, actionable insights about business situations without using risk scores or predictions."""
        
        user_prompt = scenario
        if context:
            user_prompt = f"REFERENCE CASES:\n{context}\n\nCURRENT SCENARIO:\n{scenario}"
        
        user_prompt += """

TASK:
Analyze the situation and provide:
1. Core failure dynamics
2. What actions could realistically work
3. What actions will not work and why
4. Final judgment"""
        
        # Format with chat template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        # Generate
        print("🤔 Generating response...")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return {
            'analysis': generated_text,
            'retrieved_docs': retrieved_docs if use_rag else [],
            'rag_used': use_rag
        }
    
    def launch_interface(self):
        """Launch Gradio UI"""
        
        def analyze(scenario, use_rag, temperature, max_tokens):
            try:
                result = self.generate_response(
                    scenario=scenario,
                    use_rag=use_rag,
                    temperature=temperature,
                    max_new_tokens=max_tokens
                )
                
                output = f"## ANALYSIS\n\n{result['analysis']}\n\n"
                
                if result['rag_used'] and result['retrieved_docs']:
                    output += "---\n\n## RETRIEVED REFERENCE CASES\n\n"
                    for i, doc in enumerate(result['retrieved_docs']):
                        output += f"**Case {i+1}** (relevance: {doc['score']:.3f}):\n"
                        output += f"{doc['text'][:300]}...\n\n"
                
                return output
            except Exception as e:
                return f"❌ Error: {str(e)}"
        
        interface = gr.Interface(
            fn=analyze,
            inputs=[
                gr.Textbox(
                    label="Business Scenario",
                    placeholder="Describe the company situation...",
                    lines=6
                ),
                gr.Checkbox(label="Use RAG (Retrieve similar cases)", value=True),
                gr.Slider(0.1, 1.0, value=0.7, label="Temperature"),
                gr.Slider(256, 2048, value=1024, step=128, label="Max Tokens")
            ],
            outputs=gr.Markdown(label="Judgment Analysis"),
            title="🎯 Business Judgment System",
            description="Analyze business situations and receive strategic judgment based on your fine-tuned Qwen2.5-1.5B model",
            examples=[
                [
                    "A fintech startup that raised $50M but failed to achieve product-market fit after 3 years. The company faced intense competition from established banks and struggled with customer acquisition costs that exceeded lifetime value. After burning through capital, they were forced to shut down.",
                    True,
                    0.7,
                    1024
                ],
                [
                    "Analysis of a retail company over three years shows: Year 1 - positive net income of $5M with revenue of $50M, Year 2 - break-even with revenue of $42M, Year 3 - loss of $8M with revenue of $35M. Total liabilities increased from $20M to $45M while current assets declined from $15M to $6M.",
                    True,
                    0.7,
                    1024
                ],
                [
                    "A SaaS company in the HR space operated for 5 years and raised $30M in funding. They have 200 enterprise clients but monthly burn is $2M. Customer churn is 25% annually and the average sales cycle is 9 months. Two major competitors with 10x resources have launched similar products.",
                    True,
                    0.7,
                    1024
                ]
            ],
            theme=gr.themes.Soft()
        )
        
        interface.launch(share=False, server_name="127.0.0.1", server_port=7860)


def main():
    """Main execution"""
    print("=" * 70)
    print("💬 BUSINESS JUDGMENT SYSTEM - LOCAL INFERENCE")
    print("=" * 70)
    
    try:
        # Initialize system
        system = BusinessJudgmentSystem()
        
        # Test inference
        print("\n" + "=" * 70)
        print("🧪 RUNNING TEST INFERENCE")
        print("=" * 70)
        
        test_scenario = """A SaaS startup in the HR tech space raised $30M in Series A funding. 
After two years, the company has 200 enterprise clients but is burning $2M monthly. 
Customer churn is 25% annually, and the sales cycle averages 9 months. 
Two major competitors with 10x more resources have launched similar products."""
        
        print(f"\nSCENARIO:\n{test_scenario}\n")
        print("=" * 70)
        
        result = system.generate_response(test_scenario, use_rag=True, max_new_tokens=512)
        
        print("\n📊 ANALYSIS:")
        print(result['analysis'])
        print("\n" + "=" * 70)
        
        if result['retrieved_docs']:
            print("\n📚 RETRIEVED CASES:")
            for i, doc in enumerate(result['retrieved_docs']):
                print(f"\n{i+1}. (Relevance Score: {doc['score']:.3f})")
                print(f"   {doc['text'][:150]}...")
        
        # Launch UI
        print("\n" + "=" * 70)
        print("🚀 LAUNCHING WEB INTERFACE")
        print("=" * 70)
        print("📱 Access at: http://127.0.0.1:7860")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 70)
        
        system.launch_interface()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\n💡 Troubleshooting:")
        print("   1. Check model files exist: ls -la ../models/qwen-business-judgment/")
        print("   2. Check RAG index exists: ls -la ../rag_index/")
        print("   3. Verify you're in the scripts/ directory")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()