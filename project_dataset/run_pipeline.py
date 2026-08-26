#!/usr/bin/env python
"""
Dataset Preparation & Verification Pipeline
Phases 0-4 for: Paraphrase-aware robust detection of AI-generated text using graph-based structural evidence
"""

import os
import sys
import yaml
import json
import hashlib
from datetime import datetime
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets


class PipelineLogger:
    """Logger that writes to both console and log file."""
    
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file = open(self.log_path, 'w')
        
    def log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] [{level}] {message}"
        print(line)
        self.log_file.write(line + "\n")
        self.log_file.flush()
        
    def close(self):
        self.log_file.close()


class DatasetPipeline:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.logger = PipelineLogger("project_dataset/reports/pipeline.log")
        self.audit_data = {}
        self.results = {}
        
    def log(self, msg, level="INFO"):
        self.logger.log(msg, level)
        
    def phase_0_freeze_requirements(self):
        """Phase 0: Requirements already frozen in config. Just validate."""
        self.log("=" * 60)
        self.log("PHASE 0: Freeze Dataset Requirements")
        self.log("=" * 60)
        
        cfg = self.config['dataset']
        self.log(f"Source dataset: {cfg['source']}")
        self.log(f"Human samples required: {cfg['human_samples']}")
        self.log(f"AI samples required: {cfg['ai_samples']}")
        self.log(f"Random seed: {cfg['random_seed']}")
        self.log(f"Text column: {cfg['text_column']}")
        self.log(f"Label column: {cfg['label_column']}")
        self.log(f"Label mapping: {cfg['label_mapping']}")
        self.log(f"Splits to use: {cfg['splits_to_use']}")
        
        # Verify we have enough data by checking dataset info
        self.log("Phase 0 complete - requirements documented.")
        
    def phase_1_download_audit(self):
        """Phase 1: Download and audit source dataset."""
        self.log("=" * 60)
        self.log("PHASE 1: Download and Audit Source Dataset")
        self.log("=" * 60)
        
        cfg = self.config['dataset']
        source = cfg['source']
        text_col = cfg['text_column']
        label_col = cfg['label_column']
        label_map = cfg['label_mapping']
        splits = cfg['splits_to_use']
        
        # Load dataset
        self.log(f"Loading dataset: {source}")
        ds = load_dataset(source, download_mode='reuse_cache_if_exists')
        
        # Combine splits
        combined = concatenate_datasets([ds[split] for split in splits if split in ds])
        self.log(f"Combined dataset size: {len(combined)}")
        
        # Schema info
        self.log(f"Features: {combined.features}")
        self.log(f"Columns: {combined.column_names}")
        
        # Label distribution
        labels = combined[label_col]
        dist = Counter(labels)
        self.log(f"Label distribution: {dict(dist)}")
        
        # Verify label mapping
        self.log(f"Verified label mapping: {label_map}")
        human_label = [k for k, v in label_map.items() if v == 'human'][0]
        ai_label = [k for k, v in label_map.items() if v == 'ai'][0]
        self.log(f"Human label value: {human_label}")
        self.log(f"AI label value: {ai_label}")
        
        # Count by label
        human_count = dist.get(human_label, 0)
        ai_count = dist.get(ai_label, 0)
        self.log(f"Total human records: {human_count}")
        self.log(f"Total AI records: {ai_count}")
        
        # Quality checks
        texts = combined[text_col]
        
        # Missing/None
        missing_count = sum(1 for t in texts if t is None)
        self.log(f"Missing text values: {missing_count}")
        
        # Empty strings
        empty_count = sum(1 for t in texts if t is not None and len(str(t).strip()) == 0)
        self.log(f"Empty text strings: {empty_count}")
        
        # Text length stats
        text_lengths = [len(str(t)) for t in texts if t is not None]
        self.log(f"Text length - min: {min(text_lengths)}, max: {max(text_lengths)}, mean: {np.mean(text_lengths):.1f}, median: {np.median(text_lengths):.1f}")
        
        # Short texts
        min_len = self.config['quality'].get('min_text_length', 10)
        short_count = sum(1 for l in text_lengths if l < min_len)
        self.log(f"Texts shorter than {min_len} chars: {short_count}")
        
        # Long texts
        max_len = self.config['quality'].get('max_text_length', 100000)
        long_count = sum(1 for l in text_lengths if l > max_len)
        self.log(f"Texts longer than {max_len} chars: {long_count}")
        
        # Duplicates
        unique_texts = set(str(t) for t in texts if t is not None)
        dup_count = len(texts) - len(unique_texts)
        self.log(f"Duplicate texts: {dup_count}")
        
        # Null labels
        null_labels = sum(1 for l in labels if l is None)
        self.log(f"Null labels: {null_labels}")
        
        # Invalid labels
        valid_labels = set(label_map.keys())
        invalid_labels = [l for l in labels if l not in valid_labels]
        self.log(f"Invalid labels: {len(invalid_labels)}")
        
        # Store audit data
        self.audit_data = {
            'source_dataset': source,
            'total_records': len(combined),
            'human_count': human_count,
            'ai_count': ai_count,
            'label_distribution': dict(dist),
            'label_mapping': label_map,
            'missing_text': missing_count,
            'empty_text': empty_count,
            'duplicate_texts': dup_count,
            'text_length_stats': {
                'min': min(text_lengths),
                'max': max(text_lengths),
                'mean': float(np.mean(text_lengths)),
                'median': float(np.median(text_lengths))
            },
            'short_texts': short_count,
            'long_texts': long_count,
            'null_labels': null_labels,
            'invalid_labels': len(invalid_labels),
            'splits_used': splits,
            'dataset_revision': getattr(combined, '_fingerprint', 'unknown') if hasattr(combined, '_fingerprint') else 'unknown'
        }
        
        # Write audit report
        audit_path = Path("project_dataset/reports/source_dataset_audit.md")
        with open(audit_path, 'w', encoding='utf-8') as f:
            f.write("# Source Dataset Audit Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Dataset Information\n")
            f.write(f"- **Source:** {source}\n")
            f.write(f"- **Splits Used:** {', '.join(splits)}\n")
            f.write(f"- **Total Records:** {len(combined):,}\n")
            f.write(f"- **Human Records (label {human_label}):** {human_count:,}\n")
            f.write(f"- **AI Records (label {ai_label}):** {ai_count:,}\n\n")
            f.write(f"## Label Mapping (Verified)\n")
            for k, v in label_map.items():
                f.write(f"- **{k}**: {v}\n")
            f.write(f"\n## Quality Assessment\n")
            f.write(f"- **Missing Text Values:** {missing_count}\n")
            f.write(f"- **Empty Text Strings:** {empty_count}\n")
            f.write(f"- **Duplicate Texts:** {dup_count:,}\n")
            f.write(f"- **Text Length Stats:**\n")
            f.write(f"  - Min: {min(text_lengths)} chars\n")
            f.write(f"  - Max: {max(text_lengths)} chars\n")
            f.write(f"  - Mean: {np.mean(text_lengths):.1f} chars\n")
            f.write(f"  - Median: {np.median(text_lengths):.1f} chars\n")
            f.write(f"- **Short Texts (<{min_len} chars):** {short_count}\n")
            f.write(f"- **Long Texts (>{max_len} chars):** {long_count}\n")
            f.write(f"- **Null Labels:** {null_labels}\n")
            f.write(f"- **Invalid Labels:** {len(invalid_labels)}\n")
            f.write(f"\n## Label Distribution\n")
            for label, count in sorted(dist.items()):
                label_name = label_map.get(label, f"unknown_{label}")
                f.write(f"- **{label_name} (label {label}):** {count:,}\n")
        
        self.log(f"Audit report written to: {audit_path}")
        
        # Store combined dataset for next phases
        self.combined_dataset = combined
        self.human_label = human_label
        self.ai_label = ai_label
        self.text_col = text_col
        self.label_col = label_col
        
    def phase_2_clean_prepare(self):
        """Phase 2: Clean and prepare candidate pools."""
        self.log("=" * 60)
        self.log("PHASE 2: Clean and Prepare Candidate Pools")
        self.log("=" * 60)
        
        cfg = self.config['quality']
        ds = self.combined_dataset
        text_col = self.text_col
        label_col = self.label_col
        human_label = self.human_label
        ai_label = self.ai_label
        
        # Convert to pandas for easier manipulation
        df = ds.to_pandas()
        initial_total = len(df)
        self.log(f"Initial total records: {initial_total}")
        
        # Add source provenance fields
        df['source_dataset'] = self.config['dataset']['source']
        df['source_split'] = df.get('__split', 'unknown')  # will need to track this
        df['source_row_id'] = range(len(df))
        df['original_label'] = df[label_col]
        
        # We need to track original split - let's reload with split info
        # For now, assign based on index ranges from original splits
        # Actually, let's do this properly by keeping split info
        
        # Re-load with split tracking
        source = self.config['dataset']['source']
        splits = self.config['dataset']['splits_to_use']
        ds_dict = load_dataset(source, download_mode='reuse_cache_if_exists')
        
        dfs = []
        row_offset = 0
        for split_name in splits:
            if split_name in ds_dict:
                split_ds = ds_dict[split_name]
                split_df = split_ds.to_pandas()
                split_df['source_split'] = split_name
                split_df['source_row_id'] = range(row_offset, row_offset + len(split_df))
                split_df['source_dataset'] = source
                split_df['original_label'] = split_df[label_col]
                dfs.append(split_df)
                row_offset += len(split_df)
        
        df = pd.concat(dfs, ignore_index=True)
        self.log(f"Combined with split tracking: {len(df)} records")
        
        # Track removals
        removals = {'human': Counter(), 'ai': Counter()}
        
        # Split by label
        human_df = df[df[label_col] == human_label].copy()
        ai_df = df[df[label_col] == ai_label].copy()
        
        self.log(f"Initial human: {len(human_df)}, Initial AI: {len(ai_df)}")
        
        # 1. Remove missing text
        if cfg['remove_missing']:
            for label_name, sub_df in [('human', human_df), ('ai', ai_df)]:
                before = len(sub_df)
                sub_df = sub_df[sub_df[text_col].notna()].copy()
                removed = before - len(sub_df)
                removals[label_name]['missing'] = removed
                self.log(f"{label_name}: removed {removed} missing text records")
            human_df = sub_df if label_name == 'human' else human_df
            # Re-assign correctly
            human_df = human_df[human_df[text_col].notna()].copy()
            ai_df = ai_df[ai_df[text_col].notna()].copy()
        
        # 2. Remove empty text
        if cfg['remove_empty']:
            for label_name, sub_df in [('human', human_df), ('ai', ai_df)]:
                before = len(sub_df)
                sub_df = sub_df[sub_df[text_col].astype(str).str.strip() != ''].copy()
                removed = before - len(sub_df)
                removals[label_name]['empty'] = removed
                self.log(f"{label_name}: removed {removed} empty text records")
            human_df = human_df[human_df[text_col].astype(str).str.strip() != ''].copy()
            ai_df = ai_df[ai_df[text_col].astype(str).str.strip() != ''].copy()
        
        # 3. Remove duplicates (by text content)
        if cfg['remove_duplicates']:
            for label_name, sub_df in [('human', human_df), ('ai', ai_df)]:
                before = len(sub_df)
                sub_df = sub_df.drop_duplicates(subset=[text_col]).copy()
                removed = before - len(sub_df)
                removals[label_name]['duplicates'] = removed
                self.log(f"{label_name}: removed {removed} duplicate texts")
            human_df = human_df.drop_duplicates(subset=[text_col]).copy()
            ai_df = ai_df.drop_duplicates(subset=[text_col]).copy()
        
        # 4. Length filters
        min_len = cfg.get('min_text_length', 10)
        max_len = cfg.get('max_text_length', 100000)
        for label_name, sub_df in [('human', human_df), ('ai', ai_df)]:
            before = len(sub_df)
            text_lens = sub_df[text_col].astype(str).str.len()
            sub_df = sub_df[(text_lens >= min_len) & (text_lens <= max_len)].copy()
            removed = before - len(sub_df)
            removals[label_name]['length'] = removed
            self.log(f"{label_name}: removed {removed} texts outside length [{min_len}, {max_len}]")
        human_df = human_df[(human_df[text_col].astype(str).str.len() >= min_len) & 
                           (human_df[text_col].astype(str).str.len() <= max_len)].copy()
        ai_df = ai_df[(ai_df[text_col].astype(str).str.len() >= min_len) & 
                      (ai_df[text_col].astype(str).str.len() <= max_len)].copy()
        
        self.log(f"Final human candidates: {len(human_df)}")
        self.log(f"Final AI candidates: {len(ai_df)}")
        
        # Save candidate pools
        out_fmt = self.config['output']['format_primary']
        compression = self.config['output'].get('compression', 'snappy')
        
        human_path = Path("project_dataset/processed/human_candidates.parquet")
        ai_path = Path("project_dataset/processed/ai_candidates.parquet")
        
        human_df.to_parquet(human_path, compression=compression, index=False)
        ai_df.to_parquet(ai_path, compression=compression, index=False)
        
        self.log(f"Human candidates saved: {human_path}")
        self.log(f"AI candidates saved: {ai_path}")
        
        self.results['phase2'] = {
            'human_initial': self.audit_data['human_count'],
            'ai_initial': self.audit_data['ai_count'],
            'human_final': len(human_df),
            'ai_final': len(ai_df),
            'removals': {k: dict(v) for k, v in removals.items()},
            'human_path': str(human_path),
            'ai_path': str(ai_path)
        }
        
        self.human_candidates = human_df
        self.ai_candidates = ai_df
        
    def phase_3_random_sample(self):
        """Phase 3: Randomly select 5,000 human + 5,000 AI."""
        self.log("=" * 60)
        self.log("PHASE 3: Random Sampling")
        self.log("=" * 60)
        
        cfg = self.config['dataset']
        seed = cfg['random_seed']
        n_human = cfg['human_samples']
        n_ai = cfg['ai_samples']
        
        # Verify we have enough candidates
        if len(self.human_candidates) < n_human:
            raise ValueError(f"Not enough human candidates: {len(self.human_candidates)} < {n_human}")
        if len(self.ai_candidates) < n_ai:
            raise ValueError(f"Not enough AI candidates: {len(self.ai_candidates)} < {n_ai}")
            
        # Sample with fixed seed
        self.log(f"Sampling {n_human} human, {n_ai} AI with seed {seed}")
        
        human_sampled = self.human_candidates.sample(n=n_human, random_state=seed).copy()
        ai_sampled = self.ai_candidates.sample(n=n_ai, random_state=seed).copy()
        
        # Assign stable IDs
        human_sampled['sample_id'] = [f"H{i+1:05d}" for i in range(len(human_sampled))]
        ai_sampled['sample_id'] = [f"A{i+1:05d}" for i in range(len(ai_sampled))]
        
        # Add text_type
        human_sampled['text_type'] = 'human'
        ai_sampled['text_type'] = 'ai_original'
        
        # Add provenance fields
        for df in [human_sampled, ai_sampled]:
            df['random_seed'] = seed
            df['sampling_timestamp'] = datetime.now().isoformat()
            df['label'] = df['original_label']  # verified label (same as original)
            if self.config['provenance'].get('add_word_count', True):
                df['word_count'] = df[self.text_col].astype(str).apply(lambda x: len(x.split()))
            if self.config['provenance'].get('add_character_count', True):
                df['character_count'] = df[self.text_col].astype(str).str.len()
            # Future DIPPER fields (null for now)
            df['parent_id'] = None
            df['paraphrase_level'] = None
            df['generation_model'] = None
            df['generation_model_revision'] = None
            df['lex_diversity'] = None
            df['order_diversity'] = None
            
        # Reorder columns
        cols_order = [
            'sample_id', 'source_dataset', 'source_split', 'source_row_id',
            'text', 'original_label', 'label', 'text_type',
            'word_count', 'character_count', 'random_seed', 'sampling_timestamp',
            'parent_id', 'paraphrase_level', 'generation_model',
            'generation_model_revision', 'lex_diversity', 'order_diversity'
        ]
        
        human_sampled = human_sampled[cols_order]
        ai_sampled = ai_sampled[cols_order]
        
        # Save sampled datasets
        out_fmt = self.config['output']['format_primary']
        compression = self.config['output'].get('compression', 'snappy')
        
        human_path = Path("project_dataset/sampled/human_5000.parquet")
        ai_path = Path("project_dataset/sampled/ai_5000.parquet")
        
        human_sampled.to_parquet(human_path, compression=compression, index=False)
        ai_sampled.to_parquet(ai_path, compression=compression, index=False)
        
        self.log(f"Human sampled saved: {human_path} ({len(human_sampled)} records)")
        self.log(f"AI sampled saved: {ai_path} ({len(ai_sampled)} records)")
        
        self.results['phase3'] = {
            'human_sampled': len(human_sampled),
            'ai_sampled': len(ai_sampled),
            'seed': seed,
            'human_path': str(human_path),
            'ai_path': str(ai_path),
            'human_source_ids': human_sampled['source_row_id'].tolist(),
            'ai_source_ids': ai_sampled['source_row_id'].tolist()
        }
        
        self.human_sampled = human_sampled
        self.ai_sampled = ai_sampled
        
    def phase_4_validate_freeze_export(self):
        """Phase 4: Validate, freeze, and export the 10,000 originals."""
        self.log("=" * 60)
        self.log("PHASE 4: Validate, Freeze, and Export")
        self.log("=" * 60)
        
        cfg = self.config['dataset']
        n_human = cfg['human_samples']
        n_ai = cfg['ai_samples']
        seed = cfg['random_seed']
        
        # Combine
        combined = pd.concat([self.human_sampled, self.ai_sampled], ignore_index=True)
        self.log(f"Combined dataset: {len(combined)} records")
        
        # Validation checks
        self.log("Running validation checks...")
        
        # 1. Count validation
        assert len(combined) == n_human + n_ai, f"Total count mismatch: {len(combined)} != {n_human + n_ai}"
        human_count = len(combined[combined['text_type'] == 'human'])
        ai_count = len(combined[combined['text_type'] == 'ai_original'])
        assert human_count == n_human, f"Human count mismatch: {human_count} != {n_human}"
        assert ai_count == n_ai, f"AI count mismatch: {ai_count} != {n_ai}"
        self.log(f"[OK] Count validation: Human={human_count}, AI={ai_count}, Total={len(combined)}")
        
        # 2. Label validation
        human_label = [k for k, v in cfg['label_mapping'].items() if v == 'human'][0]
        ai_label = [k for k, v in cfg['label_mapping'].items() if v == 'ai'][0]
        
        human_labels = combined[combined['text_type'] == 'human']['original_label'].unique()
        ai_labels = combined[combined['text_type'] == 'ai_original']['original_label'].unique()
        
        assert set(human_labels) == {human_label}, f"Human label mismatch: {human_labels}"
        assert set(ai_labels) == {ai_label}, f"AI label mismatch: {ai_labels}"
        self.log(f"[OK] Label validation: Human=label {human_label}, AI=label {ai_label}")
        
        # 3. Uniqueness validation
        # Sample IDs
        assert combined['sample_id'].nunique() == len(combined), "Duplicate sample_ids found"
        self.log("[OK] Sample IDs are unique")
        
        # Source row IDs within each type
        assert combined[combined['text_type'] == 'human']['source_row_id'].nunique() == n_human, "Duplicate human source IDs"
        assert combined[combined['text_type'] == 'ai_original']['source_row_id'].nunique() == n_ai, "Duplicate AI source IDs"
        self.log("[OK] Source row IDs are unique within each type")
        
        # Text uniqueness within each set
        human_texts = combined[combined['text_type'] == 'human']['text']
        ai_texts = combined[combined['text_type'] == 'ai_original']['text']
        assert human_texts.nunique() == n_human, "Duplicate texts in human set"
        assert ai_texts.nunique() == n_ai, "Duplicate texts in AI set"
        self.log("[OK] Texts are unique within each set")
        
        # Cross-set text duplicates
        cross_dup = set(human_texts).intersection(set(ai_texts))
        if cross_dup:
            self.log(f"[WARNING] {len(cross_dup)} texts appear in both human and AI sets", "WARNING")
        else:
            self.log("[OK] No cross-set text duplicates")
        
        # 4. Provenance validation
        required_fields = ['sample_id', 'source_row_id', 'text', 'original_label', 'text_type', 'source_dataset', 'random_seed']
        for field in required_fields:
            assert field in combined.columns, f"Missing required field: {field}"
            assert combined[field].notna().all(), f"Null values in required field: {field}"
        self.log("[OK] All required provenance fields present and non-null")
        
        # 5. Reproducibility test - re-run sampling and compare
        self.log("Running reproducibility test...")
        human_resampled = self.human_candidates.sample(n=n_human, random_state=seed)
        ai_resampled = self.ai_candidates.sample(n=n_ai, random_state=seed)
        
        human_ids_match = set(human_resampled['source_row_id']) == set(self.human_sampled['source_row_id'])
        ai_ids_match = set(ai_resampled['source_row_id']) == set(self.ai_sampled['source_row_id'])
        
        if human_ids_match and ai_ids_match:
            self.log("[OK] Reproducibility test PASSED - same source IDs selected")
        else:
            self.log("[ERROR] Reproducibility test FAILED - different source IDs selected", "ERROR")
            raise RuntimeError("Sampling is not deterministic!")
        
        # Save final combined dataset
        out_fmt = self.config['output']['format_primary']
        compression = self.config['output'].get('compression', 'snappy')
        
        combined_path = Path("project_dataset/sampled/originals_10000.parquet")
        combined.to_parquet(combined_path, compression=compression, index=False)
        self.log(f"Combined dataset saved: {combined_path}")
        
        # Export CSV
        csv_path = Path("project_dataset/exports/originals_10000.csv")
        combined.to_csv(csv_path, index=False)
        self.log(f"CSV export saved: {csv_path}")
        
        # Validation report
        val_path = Path("project_dataset/reports/validation_report.md")
        with open(val_path, 'w', encoding='utf-8') as f:
            f.write("# Validation Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Count Validation\n")
            f.write(f"- Human: {human_count} (expected {n_human}) [OK]\n")
            f.write(f"- AI: {ai_count} (expected {n_ai}) [OK]\n")
            f.write(f"- Total: {len(combined)} (expected {n_human + n_ai}) [OK]\n\n")
            f.write(f"## Label Validation\n")
            f.write(f"- Human records all have label {human_label} [OK]\n")
            f.write(f"- AI records all have label {ai_label} [OK]\n\n")
            f.write(f"## Uniqueness Validation\n")
            f.write(f"- Sample IDs unique: [OK]\n")
            f.write(f"- Source row IDs unique (human): [OK]\n")
            f.write(f"- Source row IDs unique (AI): [OK]\n")
            f.write(f"- Texts unique (human): [OK]\n")
            f.write(f"- Texts unique (AI): [OK]\n")
            f.write(f"- Cross-set duplicates: {len(cross_dup)} {'[WARNING]' if cross_dup else '[OK]'}\n\n")
            f.write(f"## Provenance Validation\n")
            f.write(f"- All required fields present: [OK]\n")
            f.write(f"- No nulls in required fields: [OK]\n\n")
            f.write(f"## Reproducibility Test\n")
            f.write(f"- Seed: {seed}\n")
            f.write(f"- Human source IDs match: {'[OK]' if human_ids_match else '[FAIL]'}\n")
            f.write(f"- AI source IDs match: {'[OK]' if ai_ids_match else '[FAIL]'}\n")
            f.write(f"- Overall: {'PASSED' if human_ids_match and ai_ids_match else 'FAILED'}\n\n")
            f.write(f"## Output Files\n")
            f.write(f"- Parquet: {combined_path}\n")
            f.write(f"- CSV: {csv_path}\n")
        
        # Sampling report
        samp_path = Path("project_dataset/reports/sampling_report.md")
        with open(samp_path, 'w', encoding='utf-8') as f:
            f.write("# Sampling Report\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n\n")
            f.write(f"## Configuration\n")
            f.write(f"- Random seed: {seed}\n")
            f.write(f"- Human samples: {n_human}\n")
            f.write(f"- AI samples: {n_ai}\n")
            f.write(f"- Source dataset: {cfg['source']}\n")
            f.write(f"- Label mapping: {cfg['label_mapping']}\n\n")
            f.write(f"## Candidate Pools\n")
            f.write(f"- Human candidates: {len(self.human_candidates):,}\n")
            f.write(f"- AI candidates: {len(self.ai_candidates):,}\n\n")
            f.write(f"## Selected Samples\n")
            f.write(f"- Human selected: {len(self.human_sampled):,}\n")
            f.write(f"- AI selected: {len(self.ai_sampled):,}\n\n")
            f.write(f"## Sample ID Ranges\n")
            f.write(f"- Human: H00001 - H{len(self.human_sampled):05d}\n")
            f.write(f"- AI: A00001 - A{len(self.ai_sampled):05d}\n\n")
            f.write(f"## Reproducibility\n")
            f.write(f"- Seed: {seed}\n")
            f.write(f"- Method: pandas.DataFrame.sample with random_state\n")
            f.write(f"- Verified deterministic: Yes\n")
        
        self.results['phase4'] = {
            'total_records': len(combined),
            'human_count': human_count,
            'ai_count': ai_count,
            'cross_duplicates': len(cross_dup),
            'reproducibility_passed': human_ids_match and ai_ids_match,
            'combined_path': str(combined_path),
            'csv_path': str(csv_path)
        }
        
        # Final verification: reload from disk and re-check
        self.log("Reloading from disk for final verification...")
        reloaded = pd.read_parquet(combined_path)
        assert len(reloaded) == len(combined), "Reloaded count mismatch"
        assert reloaded['sample_id'].nunique() == len(combined), "Reloaded sample IDs not unique"
        self.log("[OK] Disk reload verification passed")
        
        self.final_dataset = combined
        
    def generate_final_report(self):
        """Generate final summary report."""
        self.log("=" * 60)
        self.log("FINAL REPORT")
        self.log("=" * 60)
        
        cfg = self.config['dataset']
        
        report = f"""
Source dataset: {cfg['source']}
Source revision/version: {self.audit_data.get('dataset_revision', 'unknown')}
Original row count: {self.audit_data['total_records']:,}

Human candidates: {self.results['phase2']['human_final']:,}
Human selected: {self.results['phase3']['human_sampled']:,}
Human removed: {self.audit_data['human_count'] - self.results['phase2']['human_final']:,}

AI candidates: {self.results['phase2']['ai_final']:,}
AI selected: {self.results['phase3']['ai_sampled']:,}
AI removed: {self.audit_data['ai_count'] - self.results['phase2']['ai_final']:,}

Final total: {self.results['phase4']['total_records']:,}
Human = {self.results['phase4']['human_count']:,}
AI = {self.results['phase4']['ai_count']:,}

Random seed: {cfg['random_seed']}
Label mapping: {cfg['label_mapping']}

Duplicates (source): {self.audit_data['duplicate_texts']:,}
Missing values (source): {self.audit_data['missing_text']:,}
Validation status: PASSED

Human dataset path: {self.results['phase3']['human_path']}
AI dataset path: {self.results['phase3']['ai_path']}
Combined dataset path: {self.results['phase4']['combined_path']}
CSV export path: {self.results['phase4']['csv_path']}

Ready for DIPPER: YES
"""
        self.log(report)
        
        # Write final summary
        summary_path = Path("project_dataset/reports/FINAL_SUMMARY.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(report.strip())
            
        self.log(f"Final summary written to: {summary_path}")
        self.log("READY FOR DIPPER")
        
        return True
    
    def run(self):
        try:
            self.phase_0_freeze_requirements()
            self.phase_1_download_audit()
            self.phase_2_clean_prepare()
            self.phase_3_random_sample()
            self.phase_4_validate_freeze_export()
            self.generate_final_report()
            return True
        except Exception as e:
            self.log(f"PIPELINE FAILED: {e}", "ERROR")
            import traceback
            self.log(traceback.format_exc(), "ERROR")
            return False
        finally:
            self.logger.close()


if __name__ == "__main__":
    pipeline = DatasetPipeline("project_dataset/configs/sampling_config.yaml")
    success = pipeline.run()
    sys.exit(0 if success else 1)