
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class DatasetProcessor:
    """Base processor"""
    
    def __init__(self, output_dir: str = "/workspace/qwen-business-judgment/data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_samples = []
        self.rag_documents = []


class StartupFailuresProcessor(DatasetProcessor):
    """Dataset 1: Startup Failures"""
    
    INDICATOR_COLUMNS = [
        'Giants', 'No Budget', 'Competition', 'Poor Market Fit',
        'Acquisition Stagnation', 'Platform Dependency', 
        'High Operational Costs', 'Monetization Failure',
        'Niche Limits', 'Execution Flaws', 'Trend Shifts',
        'Toxicity/Trust Issues', 'Regulatory Pressure', 'Overhype'
    ]
    
    def process(self, data_dir: str):
        """Process all startup CSVs - NO LIMITS"""
        data_path = Path(data_dir)
        
        if not data_path.exists():
            print(f"⚠️  Skipping: {data_dir} not found")
            return
        
        csv_files = list(data_path.glob("*.csv"))
        if len(csv_files) == 0:
            print(f"⚠️  No CSV files in {data_dir}")
            return
        
        print(f"📊 Processing {len(csv_files)} startup files...")
        
        for csv_file in tqdm(csv_files, desc="Startup files"):
            try:
                df = pd.read_csv(csv_file, encoding='utf-8')
                sector = csv_file.stem.replace('_', ' ').title()
                available_indicators = [col for col in self.INDICATOR_COLUMNS if col in df.columns]
                
                # Process ALL rows
                for idx, row in df.iterrows():
                    try:
                        scenario = self._build_narrative_scenario(row, available_indicators, sector)
                        analysis = self._build_judgment_analysis(row, available_indicators)
                        
                        self.training_samples.append({
                            'input': scenario,
                            'output': self._format_output(analysis)
                        })
                        
                        self.rag_documents.append({
                            'text': f"{scenario}\n\n{analysis['dynamics']}\n\n{analysis['judgment']}",
                            'metadata': {'source': 'startup_failures', 'sector': sector}
                        })
                    except:
                        continue
            except Exception as e:
                print(f"⚠️  Error in {csv_file.name}: {e}")
        
        print(f"✅ Processed {len(self.training_samples)} startup scenarios")
    
    def _build_narrative_scenario(self, row: pd.Series, indicators: List[str], sector: str) -> str:
        """Build narrative from raw data"""
        what_did = row.get('What They Did', 'operated in the market')
        years = row.get('Years of Operation', 'several')
        raised = row.get('How Much They Raised', 'funding')
        why_failed = row.get('Why They Failed', '')
        
        active_factors = []
        for ind in indicators:
            if pd.notna(row.get(ind)) and row.get(ind) in [1, '1', True, 'Yes']:
                active_factors.append(ind)
        
        scenario = f"A {sector.lower()} company {what_did.lower()}. "
        scenario += f"After operating for {years} years and raising {raised}, the company ceased operations. "
        
        if 'Giants' in active_factors or 'Competition' in active_factors:
            scenario += "The company faced dominant competitors with vastly superior resources and market position. "
        
        if 'Poor Market Fit' in active_factors:
            scenario += "Despite efforts, the product failed to resonate with the target customer base. "
        
        if 'Monetization Failure' in active_factors:
            scenario += "The company struggled to convert users into paying customers. "
        
        if 'No Budget' in active_factors:
            scenario += "Operating with severe capital constraints limited strategic execution. "
        
        if 'Execution Flaws' in active_factors:
            scenario += "Key operational milestones were repeatedly missed. "
        
        if 'Platform Dependency' in active_factors:
            scenario += "The business model relied heavily on a third-party platform. "
        
        if why_failed:
            scenario += f"The underlying issue: {why_failed.lower()}."
        
        return scenario.strip()
    
    def _build_judgment_analysis(self, row: pd.Series, indicators: List[str]) -> Dict[str, Any]:
        """Generate reasoning-based judgment"""
        why_failed = row.get('Why They Failed', 'multiple compounding factors')
        takeaway = row.get('Takeaway', '')
        
        active_factors = [ind for ind in indicators 
                         if pd.notna(row.get(ind)) and row.get(ind) in [1, '1', True, 'Yes']]
        
        dynamics = f"The core failure mechanism centers on {why_failed.lower()}. "
        
        if 'Giants' in active_factors or 'Competition' in active_factors:
            dynamics += "Competing against entrenched players with asymmetric resources created an insurmountable barrier. "
        
        if 'Poor Market Fit' in active_factors:
            dynamics += "The product hypothesis was invalidated by market behavior. "
        
        if 'Monetization Failure' in active_factors:
            dynamics += "User engagement did not translate to willingness-to-pay. "
        
        could_work = []
        if 'Poor Market Fit' in active_factors:
            could_work.append("Radical pivot to adjacent market segment with demonstrated demand validation")
        if 'Competition' in active_factors or 'Giants' in active_factors:
            could_work.append("Strategic retreat to defensible niche where incumbents have structural disadvantages")
        if 'Monetization Failure' in active_factors:
            could_work.append("Complete business model redesign around proven monetization vector")
        if not could_work:
            could_work.append("Fundamental reconception of value proposition based on validated customer pain points")
        
        wont_work = []
        if 'Giants' in active_factors:
            wont_work.append("Direct feature competition with incumbents: they will always match and out-spend")
        if 'Poor Market Fit' in active_factors:
            wont_work.append("Increased marketing spend: no amount of distribution fixes a product nobody wants")
        if 'Monetization Failure' in active_factors:
            wont_work.append("Growth-first strategy: scaling unprofitable unit economics accelerates failure")
        if not wont_work:
            wont_work.append("Incremental optimization: fundamental problems require structural solutions")
        
        severity = len(active_factors)
        
        if severity >= 4:
            judgment = "TERMINAL DIAGNOSIS: Multiple compounding failure modes indicate the company has passed salvageability. Orderly wind-down recommended."
        elif severity >= 2:
            judgment = "CRITICAL CONDITION: Company faces existential crisis requiring immediate, radical intervention."
        else:
            judgment = "RECOVERABLE WITH DECISIVE ACTION: Core business not yet deteriorated beyond repair."
        
        if takeaway:
            judgment += f" Strategic lesson: {takeaway}"
        
        return {
            'dynamics': dynamics,
            'could_work': could_work,
            'wont_work': wont_work,
            'judgment': judgment
        }
    
    def _format_output(self, analysis: Dict[str, Any]) -> str:
        return f"""ANALYSIS:
{analysis['dynamics']}

WHAT COULD WORK:
{chr(10).join(f"• {item}" for item in analysis['could_work'])}

WHAT WILL NOT WORK:
{chr(10).join(f"• {item}" for item in analysis['wont_work'])}

FINAL JUDGMENT:
{analysis['judgment']}"""


class USBankruptcyProcessor(DatasetProcessor):
    """Dataset 2: US Bankruptcy - Process ALL rows"""
    
    FINANCIAL_CONCEPTS = {
        'liquidity': ['X1', 'X3', 'X4'],
        'profitability': ['X5', 'X6', 'X7'],
        'leverage': ['X8', 'X12', 'X13'],
        'efficiency': ['X14', 'X15'],
        'solvency': ['X16', 'X17', 'X18']
    }
    
    def process(self, file_path: str):
        """Process ALL bankruptcy rows - NO LIMITS"""
        if not Path(file_path).exists():
            print(f"⚠️  Skipping: {file_path} not found")
            return
        
        print("📊 Processing US bankruptcy dataset (ALL ROWS)...")
        df = pd.read_csv(file_path)
        
        if 'status_label' in df.columns:
            df = df.drop(columns=['status_label'])
        
        feature_cols = [col for col in df.columns if col.startswith('X')]
        
        print(f"   Total rows to process: {len(df)}")
        
        # Process ALL rows - removed limit
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Bankruptcy"):
            try:
                scenario = self._build_financial_narrative(row, feature_cols)
                analysis = self._build_financial_judgment(row, feature_cols)
                
                self.training_samples.append({
                    'input': scenario,
                    'output': self._format_output(analysis)
                })
                
                self.rag_documents.append({
                    'text': f"{scenario}\n\n{analysis['dynamics']}\n\n{analysis['judgment']}",
                    'metadata': {'source': 'us_bankruptcy'}
                })
            except:
                continue
        
        print(f"✅ Processed {len(self.training_samples)} financial scenarios")
    
    def _build_financial_narrative(self, row: pd.Series, features: List[str]) -> str:
        """Convert to narrative"""
        year = row.get('year', 'recent fiscal period')
        
        financial_stress_indicators = []
        
        liquidity_features = [row.get(f) for f in self.FINANCIAL_CONCEPTS['liquidity'] if f in features]
        liquidity_features = [x for x in liquidity_features if pd.notna(x)]
        if liquidity_features:
            avg_liquidity = np.mean(liquidity_features)
            if avg_liquidity < 0.5:
                financial_stress_indicators.append("severe liquidity constraints")
            elif avg_liquidity < 1.0:
                financial_stress_indicators.append("working capital pressures")
        
        profit_features = [row.get(f) for f in self.FINANCIAL_CONCEPTS['profitability'] if f in features]
        profit_features = [x for x in profit_features if pd.notna(x)]
        if profit_features:
            negative_count = sum(1 for x in profit_features if x < 0)
            if negative_count >= 2:
                financial_stress_indicators.append("sustained operational losses")
            elif any(x < 0 for x in profit_features):
                financial_stress_indicators.append("negative profitability")
        
        leverage_features = [row.get(f) for f in self.FINANCIAL_CONCEPTS['leverage'] if f in features]
        leverage_features = [x for x in leverage_features if pd.notna(x)]
        if leverage_features:
            avg_leverage = np.mean(leverage_features)
            if avg_leverage > 3.0:
                financial_stress_indicators.append("excessive debt burden")
            elif avg_leverage > 1.5:
                financial_stress_indicators.append("elevated leverage")
        
        scenario = f"Financial analysis of a US corporation for fiscal year {year} reveals multiple stress indicators. "
        
        if financial_stress_indicators:
            scenario += "The company exhibits: " + "; ".join(financial_stress_indicators) + ". "
        else:
            scenario += "The company shows signs of deteriorating financial health. "
        
        scenario += "Traditional financial metrics suggest ability to meet obligations is materially compromised."
        
        return scenario
    
    def _build_financial_judgment(self, row: pd.Series, features: List[str]) -> Dict[str, Any]:
        """Generate judgment"""
        stress_count = 0
        
        for concept, feature_list in self.FINANCIAL_CONCEPTS.items():
            values = [row.get(f) for f in feature_list if f in features and pd.notna(row.get(f))]
            if values:
                if concept == 'liquidity' and np.mean(values) < 0.7:
                    stress_count += 1
                elif concept == 'profitability' and sum(1 for v in values if v < 0) >= 1:
                    stress_count += 2
                elif concept == 'leverage' and np.mean(values) > 2.0:
                    stress_count += 1
                elif concept == 'solvency' and any(v < 0 for v in values):
                    stress_count += 2
        
        dynamics = "Financial deterioration stems from operational underperformance compounded by capital structure weakness. "
        
        if stress_count >= 4:
            dynamics += "Multiple dimensions of distress indicate systemic dysfunction. "
        
        could_work = [
            "Comprehensive debt restructuring with creditor concessions",
            "Strategic divestiture of non-core assets",
            "Chapter 11 bankruptcy filing",
            "Emergency equity infusion from distressed investor"
        ]
        
        wont_work = [
            "Incremental cost reductions: structural losses require transformation",
            "Refinancing: lenders won't extend credit to near-insolvent entities",
            "Organic growth: lacks financial runway",
            "Waiting for recovery: requires immediate intervention"
        ]
        
        if stress_count >= 5:
            judgment = "IMMINENT INSOLVENCY: Bankruptcy filing highly probable within 6-12 months."
        elif stress_count >= 3:
            judgment = "SEVERE DISTRESS: Existential financial crisis. Bankruptcy likely within 12-24 months."
        else:
            judgment = "FINANCIAL STRESS: Concerning but not yet terminal. Negotiated restructuring possible."
        
        return {
            'dynamics': dynamics,
            'could_work': could_work,
            'wont_work': wont_work,
            'judgment': judgment
        }
    
    def _format_output(self, analysis: Dict[str, Any]) -> str:
        return f"""ANALYSIS:
{analysis['dynamics']}

WHAT COULD WORK:
{chr(10).join(f"• {item}" for item in analysis['could_work'])}

WHAT WILL NOT WORK:
{chr(10).join(f"• {item}" for item in analysis['wont_work'])}

FINAL JUDGMENT:
{analysis['judgment']}"""


class LongHorizonProcessor(DatasetProcessor):
    """Dataset 5: Long Horizon - Process ALL rows"""
    
    METRICS = [
        'current_assets', 'total_assets', 'total_liabilities',
        'net_income', 'ebit', 'gross_profit', 'total_revenue',
        'retained_earnings', 'operating_expenses'
    ]
    
    def process(self, file_path: str):
        """Process ALL long-horizon rows - NO LIMITS"""
        if not Path(file_path).exists():
            print(f"⚠️  Skipping: {file_path} not found")
            return
        
        print("📊 Processing long-horizon dataset (ALL ROWS)...")
        df = pd.read_csv(file_path)
        
        if 'status_label' in df.columns:
            df = df.drop(columns=['status_label'])
        
        print(f"   Total rows to process: {len(df)}")
        
        # Process ALL rows - removed limit
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Long-horizon"):
            try:
                scenario = self._build_temporal_narrative(row)
                analysis = self._build_decay_judgment(row)
                
                self.training_samples.append({
                    'input': scenario,
                    'output': self._format_output(analysis)
                })
                
                self.rag_documents.append({
                    'text': f"{scenario}\n\n{analysis['dynamics']}\n\n{analysis['judgment']}",
                    'metadata': {'source': 'long_horizon'}
                })
            except:
                continue
        
        print(f"✅ Processed {len(self.training_samples)} temporal scenarios")
    
    def _build_temporal_narrative(self, row: pd.Series) -> str:
        """Build multi-year narrative"""
        year = row.get('fyear', 'recent period')
        
        scenario = f"Three-year financial trajectory analysis for a company through fiscal year {year}. "
        
        trend_descriptions = []
        
        if all(f"{i}_total_revenue" in row.index for i in [1, 2, 3]):
            rev = [row.get(f"{i}_total_revenue") for i in [1, 2, 3]]
            if all(pd.notna(r) for r in rev):
                if rev[0] > rev[1] > rev[2] and rev[2] < rev[0] * 0.7:
                    trend_descriptions.append("revenue declining sharply year-over-year")
        
        if all(f"{i}_net_income" in row.index for i in [1, 2, 3]):
            income = [row.get(f"{i}_net_income") for i in [1, 2, 3]]
            if all(pd.notna(i) for i in income):
                if income[0] > 0 and income[2] < 0:
                    trend_descriptions.append("profitability deteriorating from positive to negative")
                elif all(i < 0 for i in income):
                    trend_descriptions.append("sustained losses across all years")
        
        if trend_descriptions:
            scenario += "Multi-year patterns reveal: " + "; ".join(trend_descriptions) + ". "
        else:
            scenario += "The company exhibits progressive financial weakening. "
        
        scenario += "Trends indicate structural deterioration rather than cyclical fluctuations."
        
        return scenario
    
    def _build_decay_judgment(self, row: pd.Series) -> Dict[str, Any]:
        """Generate judgment"""
        decay_signals = 0
        
        for metric in self.METRICS:
            if all(f"{i}_{metric}" in row.index for i in [1, 2, 3]):
                vals = [row.get(f"{i}_{metric}") for i in [1, 2, 3]]
                
                if all(pd.notna(v) for v in vals):
                    if metric in ['net_income', 'ebit', 'gross_profit']:
                        if vals[0] > 0 and vals[2] < 0:
                            decay_signals += 2
                        elif vals[0] > vals[1] > vals[2] and vals[2] < vals[0] * 0.5:
                            decay_signals += 1
                    elif metric in ['total_revenue']:
                        if vals[0] > vals[1] > vals[2] and vals[2] < vals[0] * 0.7:
                            decay_signals += 1
        
        dynamics = "Multi-year analysis reveals accelerating deterioration. "
        
        if decay_signals >= 6:
            dynamics += "Persistence of negative trends indicates failed turnaround attempts. "
        
        could_work = [
            "Immediate Chapter 11 filing",
            "Management replacement with turnaround specialists",
            "Strategic sale to competitor",
            "Rapid divestiture of unprofitable units"
        ]
        
        wont_work = [
            "Gradual turnaround: multi-year decline shows incremental approaches failed",
            "New initiatives: lacks runway and capability",
            "Market repositioning: issues are internal",
            "Cost optimization: likely already lean"
        ]
        
        if decay_signals >= 7:
            judgment = "TERMINAL TRAJECTORY: Company passed viability point. Liquidation recommended."
        elif decay_signals >= 4:
            judgment = "CRITICAL DECAY: Bankruptcy virtually certain within 12 months without radical intervention."
        else:
            judgment = "CONCERNING WEAKENING: Requires urgent strategic reset."
        
        return {
            'dynamics': dynamics,
            'could_work': could_work,
            'wont_work': wont_work,
            'judgment': judgment
        }
    
    def _format_output(self, analysis: Dict[str, Any]) -> str:
        return f"""ANALYSIS:
{analysis['dynamics']}

WHAT COULD WORK:
{chr(10).join(f"• {item}" for item in analysis['could_work'])}

WHAT WILL NOT WORK:
{chr(10).join(f"• {item}" for item in analysis['wont_work'])}

FINAL JUDGMENT:
{analysis['judgment']}"""


def main():
    """Main pipeline - processes ALL data"""
    print("=" * 70)
    print("🚀 COMPLETE DATA PREPARATION (ALL ROWS)")
    print("=" * 70)
    
    output_dir = "/workspace/qwen-business-judgment/data/processed"
    
    processors = []
    
    print("\n📁 DATASET 1: Startup Failures")
    p1 = StartupFailuresProcessor(output_dir)
    p1.process("/workspace/qwen-business-judgment/data/raw/startup_failures/")
    processors.append(p1)
    
    print("\n📁 DATASET 2: US Bankruptcy (ALL ROWS)")
    p2 = USBankruptcyProcessor(output_dir)
    p2.process("/workspace/qwen-business-judgment/data/raw/us_bankruptcy.csv")
    processors.append(p2)
    
    print("\n📁 DATASET 5: Long Horizon (ALL ROWS)")
    p3 = LongHorizonProcessor(output_dir)
    for possible_path in [
        "/workspace/qwen-business-judgment/data/raw/long_horizon/bankruptcy_data.csv",
        "/workspace/qwen-business-judgment/data/raw/long_horizon/data.csv",
    ]:
        if Path(possible_path).exists():
            p3.process(possible_path)
            break
    processors.append(p3)
    
    print("\n📦 Combining datasets...")
    all_samples = []
    all_docs = []
    
    for p in processors:
        all_samples.extend(p.training_samples)
        all_docs.extend(p.rag_documents)
    
    train_path = Path(output_dir) / "training_data.jsonl"
    with open(train_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    rag_path = Path(output_dir) / "rag_corpus.jsonl"
    with open(rag_path, 'w', encoding='utf-8') as f:
        for doc in all_docs:
            f.write(json.dumps(doc, ensure_ascii=False) + '\n')
    
    print(f"\n✅ COMPLETE")
    print(f"📊 Total training samples: {len(all_samples)}")
    print(f"📚 RAG documents: {len(all_docs)}")
    
    if all_samples:
        print("\n📄 SAMPLE:")
        sample = all_samples[0]
        print(f"INPUT: {sample['input'][:200]}...")
        print(f"OUTPUT: {sample['output'][:200]}...")


if __name__ == "__main__":
    main()
