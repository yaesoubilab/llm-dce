import pandas as pd
import numpy as np
import os
import glob
import re
import json
from collections import Counter
from typing import Dict, List, Tuple
import asyncio
import nest_asyncio
from openai import AsyncAzureOpenAI
import seaborn as sns
import matplotlib.pyplot as plt

nest_asyncio.apply()

# ================================
# PART 1: TRADITIONAL TEXT ANALYSIS
# ================================

class ReasoningAnalyzer:
    """Traditional text-based analysis of reasoning"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def compute_basic_metrics(self) -> pd.DataFrame:
        """Calculate basic quantitative metrics for reasoning text"""
        
        # Length metrics
        self.df['reasoning_length'] = self.df['reasoning'].str.len()
        self.df['reasoning_word_count'] = self.df['reasoning'].str.split().str.len()
        self.df['reasoning_sentence_count'] = self.df['reasoning'].str.count(r'[.!?]+')
        self.df['avg_word_length'] = self.df['reasoning'].apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if pd.notna(x) else 0
        )
        self.df['avg_sentence_length'] = (
            self.df['reasoning_word_count'] / self.df['reasoning_sentence_count'].replace(0, 1)
        )
        
        return self.df
    
    def compute_vocabulary_diversity(self) -> pd.DataFrame:
        """Measure lexical diversity and complexity"""
        
        def type_token_ratio(text):
            """Unique words / total words"""
            if pd.isna(text) or len(str(text).split()) == 0:
                return 0
            words = str(text).lower().split()
            return len(set(words)) / len(words)
        
        self.df['vocabulary_diversity'] = self.df['reasoning'].apply(type_token_ratio)
        
        return self.df
    
    def detect_factor_mentions(self) -> pd.DataFrame:
        """Detect mentions of key decision factors"""
        
        # Define factor keywords
        factor_keywords = {
            'infection_rate': r'\b(infection|case|spread|disease|transmission|epidemic)\b',
            'npi_measures': r'\b(restriction|lockdown|mandate|measure|policy|intervention|npi)\b',
            'economic': r'\b(economic|economy|financial|business|cost|employment|job)\b',
            'capacity': r'\b(capacity|hospital|healthcare|resource|bed|facility)\b',
            'social': r'\b(social|community|public|people|population|society)\b',
            'comparative': r'\b(compar|versus|vs|between|option|alternative|choice)\b',
            'quantitative': r'\b(\d+%|\d+\.\d+|higher|lower|more|less|reduce|increase)\b',
            'causal': r'\b(because|therefore|thus|since|due to|as a result|leads to|causes)\b'
        }
        
        for factor, pattern in factor_keywords.items():
            self.df[f'mentions_{factor}'] = self.df['reasoning'].str.contains(
                pattern, case=False, na=False, regex=True
            ).astype(int)
        
        # Total factor coverage
        mention_cols = [col for col in self.df.columns if col.startswith('mentions_')]
        self.df['total_factors_mentioned'] = self.df[mention_cols].sum(axis=1)
        
        return self.df
    
    def detect_context_awareness(self) -> pd.DataFrame:
        """Check if context variables are explicitly mentioned"""
        
        # These will vary based on your strategies
        context_patterns = {
            'capacity_mentioned': r'\b(capacity|hospital|healthcare system)\b',
            'economic_mentioned': r'\b(economic|economy|financial impact)\b',
            'preference_mentioned': r'\b(prefer|preference|priority)\b'
        }
        
        for var, pattern in context_patterns.items():
            self.df[f'context_{var}'] = self.df['reasoning'].str.contains(
                pattern, case=False, na=False, regex=True
            ).astype(int)
        
        return self.df
    
    def compute_logical_structure_metrics(self) -> pd.DataFrame:
        """Analyze logical structure of reasoning"""
        
        # Count logical connectors
        self.df['has_comparison'] = self.df['reasoning'].str.contains(
            r'\b(however|whereas|while|although|but|compare|versus)\b',
            case=False, na=False
        ).astype(int)
        
        self.df['has_causation'] = self.df['reasoning'].str.contains(
            r'\b(because|therefore|thus|hence|consequently|as a result)\b',
            case=False, na=False
        ).astype(int)
        
        self.df['has_conclusion'] = self.df['reasoning'].str.contains(
            r'\b(therefore|thus|consequently|in conclusion|overall)\b',
            case=False, na=False
        ).astype(int)
        
        return self.df
    
    def analyze_consistency_within_model(self) -> pd.DataFrame:
        """Calculate within-model consistency metrics"""
        
        # Group by model and similar contexts
        consistency_metrics = []
        
        for model in self.df['model'].unique():
            model_df = self.df[self.df['model'] == model]
            
            # Calculate variability in reasoning length
            length_cv = model_df['reasoning_word_count'].std() / model_df['reasoning_word_count'].mean()
            
            # Calculate variability in factor mentions
            factor_cols = [col for col in self.df.columns if col.startswith('mentions_')]
            factor_consistency = model_df[factor_cols].std().mean()
            
            consistency_metrics.append({
                'model': model,
                'reasoning_length_cv': length_cv,
                'factor_mention_std': factor_consistency,
                'n_samples': len(model_df)
            })
        
        return pd.DataFrame(consistency_metrics)
    
    def generate_summary_statistics(self) -> pd.DataFrame:
        """Generate summary statistics by model and context"""
        
        # Group by model
        summary = self.df.groupby('model').agg({
            'reasoning_word_count': ['mean', 'std', 'min', 'max'],
            'vocabulary_diversity': ['mean', 'std'],
            'total_factors_mentioned': ['mean', 'std'],
            'has_comparison': 'mean',
            'has_causation': 'mean',
            'has_conclusion': 'mean'
        }).round(3)
        
        return summary


# ================================
# PART 2: LLM-AS-A-JUDGE FRAMEWORK
# ================================

class LLMJudge:
    """LLM-based evaluation of reasoning quality"""
    
    def __init__(self, api_key: str, api_version: str, endpoint: str):
        self.client = AsyncAzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=endpoint
        )
        self.sem = asyncio.Semaphore(128)
        self.rate_delay = 0.1
        self.max_retries = 5
    
    def create_evaluation_prompts(self) -> Dict[str, str]:
        """
        Create separate evaluation prompts for different dimensions.
        Following best practice: use binary classification and chain-of-thought.
        """
        
        prompts = {
            'factual_accuracy': """You are evaluating the factual accuracy of reasoning provided by an LLM for a decision task.

Given the following information:
- Option Selected: {option_selected}
- NPI Level Selected: {npi_selected}
- Infection Rate Selected: {infection_rate_selected}
- NPI Level Non-Selected: {npi_non_selected}
- Infection Rate Non-Selected: {infection_rate_non_selected}
- Context: {context}

Reasoning to evaluate:
{reasoning}

Task: Evaluate whether the reasoning correctly represents the numerical values presented in the options.

First, analyze step-by-step:
1. Does the reasoning correctly identify which option has higher/lower infection rates?
2. Does the reasoning correctly identify which option has more/fewer restrictions?
3. Are the numerical comparisons accurate?

Then provide your evaluation in JSON format:
{{"analysis": "your step-by-step analysis here", "factually_accurate": true/false}}

Factually accurate means ALL numerical facts are correct. If any fact is wrong or misrepresented, mark as false.""",

            'context_awareness': """You are evaluating whether an LLM's reasoning demonstrates awareness of provided context.

Context provided: {context}
Context type: {context_type}

Reasoning to evaluate:
{reasoning}

Task: Determine if the reasoning explicitly considers or mentions the context provided.

First, analyze step-by-step:
1. Is the context variable mentioned or referenced?
2. Does the reasoning explain how context influences the decision?
3. Is the context integrated meaningfully (not just mentioned)?

Then provide your evaluation in JSON format:
{{"analysis": "your step-by-step analysis", "context_aware": true/false}}

Mark as true only if context is meaningfully integrated into the reasoning.""",

            'logical_coherence': """You are evaluating the logical coherence of reasoning.

Reasoning to evaluate:
{reasoning}

Decision made: Selected option with NPI={npi_selected}, Infection={infection_rate_selected}
            Over option with NPI={npi_non_selected}, Infection={infection_rate_non_selected}

Task: Evaluate whether the reasoning follows a logical structure and the conclusion matches the explanation.

First, analyze step-by-step:
1. Does the reasoning present clear criteria for decision-making?
2. Is there a logical flow from analysis to conclusion?
3. Does the stated reasoning actually support the choice made?
4. Are there internal contradictions?

Then provide your evaluation in JSON format:
{{"analysis": "your step-by-step analysis", "logically_coherent": true/false}}

Mark as true only if reasoning is coherent AND supports the decision made.""",

            'comparative_analysis': """You are evaluating whether reasoning includes comparative analysis.

Option A: NPI={npi_selected}, Infection={infection_rate_selected}
Option B: NPI={npi_non_selected}, Infection={infection_rate_non_selected}

Reasoning to evaluate:
{reasoning}

Task: Determine if the reasoning explicitly compares the two options.

First, analyze step-by-step:
1. Are both options mentioned or described?
2. Are trade-offs between options discussed?
3. Are specific differences highlighted?

Then provide your evaluation in JSON format:
{{"analysis": "your step-by-step analysis", "includes_comparison": true/false}}

Mark as true only if there is explicit comparison between the two options.""",

            'depth_of_analysis': """You are evaluating the depth and sophistication of reasoning.

Reasoning to evaluate:
{reasoning}

Task: Determine if the reasoning demonstrates sophisticated analysis beyond surface-level observations.

First, analyze step-by-step:
1. Does it go beyond simply stating the numbers?
2. Does it discuss implications, trade-offs, or broader considerations?
3. Does it show multi-factor thinking?
4. Is there evidence of weighing competing priorities?

Then provide your evaluation in JSON format:
{{"analysis": "your step-by-step analysis", "deep_analysis": true/false}}

Mark as true only if the analysis shows sophisticated, multi-dimensional thinking."""
        }
        
        return prompts
    
    async def evaluate_single_reasoning(
        self, 
        row: pd.Series, 
        prompt_template: str,
        model: str = "gpt-5-mini-2025-08-07",
        temperature: float = 0.2
    ) -> Dict:
        """Evaluate a single reasoning instance"""
        
        # Format the prompt with row data
        context_col = self._get_context_column(row.get('strategy', ''))
        context_value = row.get(context_col, 'None') if context_col else 'None'
        
        prompt = prompt_template.format(
            option_selected=row.get('option_selected', ''),
            npi_selected=row.get('npi_level_selected', ''),
            infection_rate_selected=row.get('infection_rate_selected', ''),
            npi_non_selected=row.get('npi_level_non_selected', ''),
            infection_rate_non_selected=row.get('infection_rate_non_selected', ''),
            context=context_value,
            context_type=context_col,
            reasoning=row.get('reasoning', '')
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        # Retry logic
        backoff = 1.0
        for attempt in range(1, self.max_retries + 1):
            try:
                async with self.sem:
                    if self.rate_delay > 0:
                        await asyncio.sleep(self.rate_delay)
                    
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        response_format={"type": "json_object"},
                        timeout=30000
                    )
                
                result = json.loads(response.choices[0].message.content)
                return result
                
            except Exception as e:
                if attempt == self.max_retries:
                    print(f"[ERROR] Row index {row.name}: {type(e).__name__}: {e}")
                    return {"analysis": "ERROR", "result": None}
                await asyncio.sleep(backoff + np.random.random() * 0.25 * backoff)
                backoff = min(backoff * 2, 16)
    
    def _get_context_column(self, strategy_name: str) -> str:
        """Extract context column based on strategy"""
        strategy_num = int(''.join(filter(str.isdigit, strategy_name))) if any(c.isdigit() for c in strategy_name) else 0
        context_mapping = {
            2: 'capacity_context',
            3: 'economic_context',
            4: 'user_preference_type'
        }
        return context_mapping.get(strategy_num, None)
    
    async def evaluate_all_dimensions(
        self, 
        df: pd.DataFrame,
        model: str = "gpt-5-mini-2025-08-07"
    ) -> pd.DataFrame:
        """Evaluate all reasoning entries across all dimensions"""
        
        prompts = self.create_evaluation_prompts()
        results = df.copy()
        
        for dimension, prompt_template in prompts.items():
            print(f"\nEvaluating dimension: {dimension}")
            
            tasks = [
                self.evaluate_single_reasoning(row, prompt_template, model)
                for _, row in df.iterrows()
            ]
            
            dimension_results = await asyncio.gather(*tasks)
            
            # Extract results
            analyses = [r.get('analysis', '') for r in dimension_results]
            scores = [r.get(list(r.keys())[-1], None) for r in dimension_results if 'analysis' in r]
            
            results[f'{dimension}_analysis'] = analyses
            results[f'{dimension}_score'] = scores
        
        return results


# ================================
# PART 3: INTEGRATED ANALYSIS PIPELINE
# ================================

class IntegratedReasoningAnalysis:
    """Combined traditional and LLM-based analysis"""
    
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df
        self.traditional_analyzer = ReasoningAnalyzer(results_df)
        
    def run_traditional_analysis(self) -> pd.DataFrame:
        """Run all traditional text analysis"""
        
        print("Running traditional text analysis...")
        self.df = self.traditional_analyzer.compute_basic_metrics()
        self.df = self.traditional_analyzer.compute_vocabulary_diversity()
        self.df = self.traditional_analyzer.detect_factor_mentions()
        self.df = self.traditional_analyzer.detect_context_awareness()
        self.df = self.traditional_analyzer.compute_logical_structure_metrics()
        
        print("Traditional analysis complete.")
        return self.df
    
    async def run_llm_judge_analysis(
        self,
        api_key: str,
        api_version: str,
        endpoint: str,
        model: str = "gpt-5-mini-2025-08-07"
    ) -> pd.DataFrame:
        """Run LLM-as-a-judge evaluation"""
        
        print("\nRunning LLM-as-a-judge evaluation...")
        judge = LLMJudge(api_key, api_version, endpoint)
        self.df = await judge.evaluate_all_dimensions(self.df, model)
        
        print("LLM judge evaluation complete.")
        return self.df
    
    def generate_model_comparison_report(self) -> pd.DataFrame:
        """Generate comprehensive comparison across models"""
        
        # Traditional metrics by model
        traditional_cols = [
            'reasoning_word_count', 'vocabulary_diversity',
            'total_factors_mentioned', 'has_comparison',
            'has_causation', 'has_conclusion'
        ]
        
        # LLM judge metrics by model
        judge_cols = [col for col in self.df.columns if col.endswith('_score')]
        
        comparison = self.df.groupby('model').agg({
            **{col: 'mean' for col in traditional_cols if col in self.df.columns},
            **{col: lambda x: x.sum() / len(x) for col in judge_cols if col in self.df.columns}
        }).round(3)
        
        return comparison
    
    def generate_context_comparison_report(self, context_col: str) -> pd.DataFrame:
        """Generate comparison across different contexts"""
        
        if context_col not in self.df.columns:
            print(f"Warning: {context_col} not found in dataframe")
            return pd.DataFrame()
        
        comparison = self.df.groupby([' model', context_col]).agg({
            'reasoning_word_count': 'mean',
            'total_factors_mentioned': 'mean',
            'context_awareness_score': lambda x: x.sum() / len(x) if 'context_awareness_score' in self.df.columns else 0
        }).round(3)
        
        return comparison
    
    def save_results(self, output_dir: str = './reasoning_analysis_results'):
        """Save all analysis results"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save full analyzed dataset
        self.df.to_csv(f'{output_dir}/full_analysis.csv', index=False)
        
        # Save model comparison
        model_comparison = self.generate_model_comparison_report()
        model_comparison.to_csv(f'{output_dir}/model_comparison.csv')
        
        # Save consistency metrics
        consistency = self.traditional_analyzer.analyze_consistency_within_model()
        consistency.to_csv(f'{output_dir}/model_consistency.csv', index=False)
        
        print(f"\nResults saved to {output_dir}/")
        
        return output_dir


# ================================
# PART 4: USAGE EXAMPLE
# ================================

def main():
    # Load your data (modify as needed)
    strategies = {
        'strategy_one': 'strategy_one_results',
        'strategy_two': 'strategy_two_results',
        'strategy_three': 'strategy_three_results',
        'strategy_four': 'strategy_four_results'
    }
    
    all_dfs = []
    for strategy_name, directory in strategies.items():
        dir_path = os.path.join('.', directory)
        if not os.path.exists(dir_path):
            continue
        csv_files = glob.glob(os.path.join(dir_path, 'results_*.csv'))
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            df['strategy'] = strategy_name
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data found")
        return
    
    results_df = pd.concat(all_dfs, ignore_index=True)
    
    # Initialize analysis
    analyzer = IntegratedReasoningAnalysis(results_df)
    
    # Run traditional analysis
    analyzed_df = analyzer.run_traditional_analysis()
    
    # Run LLM-as-a-judge analysis
    API_KEY = 'MmI5OWMyOWRhMmFhNGFkNTljNDhjNTEzMTIwZWRjNzM6NTlhYjI3ZDdiQzM1NEFiMkI5MjNmRTcwOUM0QUYwZEQ='
    API_VERSION = '2024-10-21'
    ENDPOINT = 'https://unified-api.ucsf.edu/general'
    
    analyzed_df = asyncio.run(
        analyzer.run_llm_judge_analysis(API_KEY, API_VERSION, ENDPOINT)
    )
    
    # Generate reports
    model_comparison = analyzer.generate_model_comparison_report()
    print("\n=== Model Comparison ===")
    print(model_comparison)
    
    # Save all results
    analyzer.save_results()
    
    return analyzed_df, model_comparison

if __name__ == "__main__":
    results, comparison = main()
