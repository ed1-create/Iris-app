"""
Helper functions for Patient Segmentation dashboard
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List


def load_segmentation_data(cohort: str) -> Dict:
    """
    Load cluster profiles, assignments, and evaluation data for a cohort
    
    Args:
        cohort: 'i10' or 'z01'
    
    Returns:
        Dictionary with keys: profiles, assignments, evaluation, medoids
    """
    base_path = Path("notebooks/outputs/data")
    cohort_lower = cohort.lower()
    
    data = {}
    
    try:
        data['profiles'] = pd.read_csv(base_path / f"{cohort_lower}_cluster_profiles.csv")
        data['assignments'] = pd.read_csv(base_path / f"{cohort_lower}_cluster_assignments.csv")
        data['evaluation'] = pd.read_csv(base_path / f"{cohort_lower}_clustering_evaluation.csv")
        data['medoids'] = pd.read_csv(base_path / f"{cohort_lower}_cluster_medoids.csv")
    except Exception as e:
        raise FileNotFoundError(f"Error loading data for {cohort}: {e}")
    
    return data


def get_bmi_category_from_bmi_class(cohort: str, cluster_id: int) -> str:
    """
    Get BMI category from bmi_class mode in assignments (for Z01)
    Uses the same logic as generate_z01_report.py
    
    Args:
        cohort: 'i10' or 'z01'
        cluster_id: Cluster number
    
    Returns:
        BMI category string (e.g., 'Normal-Wt', 'Overweight', 'Obese-I', 'BMI-Unknown')
    """
    if cohort.lower() != 'z01':
        return None
    
    data = load_segmentation_data(cohort)
    assignments = data['assignments']
    
    cluster_assignments = assignments[assignments['cluster'] == cluster_id]
    
    if len(cluster_assignments) == 0 or 'bmi_class' not in cluster_assignments.columns:
        return "BMI-Unknown"
    
    bmi_class_mode = cluster_assignments['bmi_class'].mode()
    
    if len(bmi_class_mode) == 0:
        return "BMI-Unknown"
    
    bmi_class_val = bmi_class_mode.iloc[0]
    
    bmi_map = {
        'Normal': 'Normal-Wt',
        'Overweight': 'Overweight',
        'Obesity I': 'Obese-I',
        'Obesity II+': 'Obese-II+',
        'Missing': 'BMI-Unknown'
    }
    
    return bmi_map.get(str(bmi_class_val), 'BMI-Unknown')


def get_cluster_summary(cohort: str, cluster_id: int) -> Dict:
    """
    Get detailed cluster information
    
    Args:
        cohort: 'i10' or 'z01'
        cluster_id: Cluster number
    
    Returns:
        Dictionary with cluster details (includes bmi_category for Z01)
    """
    data = load_segmentation_data(cohort)
    profiles = data['profiles']
    
    cluster_data = profiles[profiles['cluster'] == cluster_id]
    
    if len(cluster_data) == 0:
        return {}
    
    result = cluster_data.iloc[0].to_dict()
    
    # For Z01, add BMI category from bmi_class mode
    if cohort.lower() == 'z01':
        bmi_category = get_bmi_category_from_bmi_class(cohort, cluster_id)
        result['bmi_category'] = bmi_category
    
    return result


def get_patient_cluster(patient_id: str, cohort: str) -> Optional[int]:
    """
    Lookup patient cluster assignment
    
    Args:
        patient_id: Patient ID
        cohort: 'i10' or 'z01'
    
    Returns:
        Cluster ID or None if not found
    """
    data = load_segmentation_data(cohort)
    assignments = data['assignments']
    
    patient_row = assignments[assignments['pid'] == patient_id]
    
    if len(patient_row) == 0:
        return None
    
    return int(patient_row.iloc[0]['cluster'])


def get_evaluation_metrics(cohort: str) -> Dict:
    """
    Get clustering evaluation metrics
    
    Args:
        cohort: 'i10' or 'z01'
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Normalize cohort to ensure consistent file loading
    cohort = cohort.lower().strip()
    
    base_path = Path("notebooks/outputs/data")
    eval_file = base_path / f"{cohort}_clustering_evaluation.csv"
    
    # Verify file exists
    if not eval_file.exists():
        raise FileNotFoundError(f"Evaluation file not found: {eval_file}")
    
    # Load evaluation data directly to ensure we get the right file
    eval_df = pd.read_csv(eval_file)
    
    if len(eval_df) == 0:
        raise ValueError(f"Evaluation file is empty: {eval_file}")
    
    return {
        'optimal_k': int(eval_df['optimal_k'].iloc[0]),
        'silhouette_score': float(eval_df['silhouette_score'].iloc[0]),
        'stability_jaccard_mean': float(eval_df['stability_jaccard_mean'].iloc[0]),
        'n_patients': int(eval_df['n_patients'].iloc[0])
    }


def get_visualization_path(cohort: str, viz_name: str) -> Optional[Path]:
    """
    Get path to visualization image
    
    Args:
        cohort: 'i10' or 'z01'
        viz_name: Name of visualization (e.g., 'silhouette_comparison', 'pca_visualization_beautiful_k4')
    
    Returns:
        Path to image or None if not found
    """
    base_path = Path("notebooks/outputs/visualizations")
    cohort_lower = cohort.lower()
    viz_path = base_path / f"{cohort_lower}_clustering" / f"{viz_name}.png"
    
    if viz_path.exists():
        return viz_path
    return None


def get_available_visualizations(cohort: str) -> list:
    """
    Get list of available visualizations for a cohort
    
    Args:
        cohort: 'i10' or 'z01'
    
    Returns:
        List of visualization names
    """
    base_path = Path("notebooks/outputs/visualizations")
    cohort_lower = cohort.lower()
    viz_dir = base_path / f"{cohort_lower}_clustering"
    
    if not viz_dir.exists():
        return []
    
    viz_files = list(viz_dir.glob("*.png"))
    return [f.stem for f in viz_files]


def create_cluster_comparison_data(cohort: str) -> pd.DataFrame:
    """
    Create comparison data for all clusters
    
    Args:
        cohort: 'i10' or 'z01'
    
    Returns:
        DataFrame with key metrics for comparison
    """
    data = load_segmentation_data(cohort)
    profiles = data['profiles']
    
    # Select key columns for comparison
    key_cols = ['cluster', 'n_patients', 'pct_patients', 'cluster_name']
    
    if 'sbp_latest_median' in profiles.columns:
        key_cols.extend(['sbp_latest_median', 'age_median'])
    if 'bmi_latest_median' in profiles.columns:
        key_cols.extend(['bmi_latest_median'])
    if 'encounter_count_12m_median' in profiles.columns:
        key_cols.extend(['encounter_count_12m_median'])
    if 'icd3_count_median' in profiles.columns:
        key_cols.extend(['icd3_count_median'])
    
    # Get available columns
    available_cols = [col for col in key_cols if col in profiles.columns]
    
    return profiles[available_cols].copy()


def get_patient_list(cohort: str) -> List[str]:
    """
    Get list of all patient IDs for a cohort
    
    Args:
        cohort: 'i10' or 'z01'
    
    Returns:
        List of patient IDs
    """
    data = load_segmentation_data(cohort)
    assignments = data['assignments']
    return sorted(assignments['pid'].unique().tolist())


def get_patient_details(patient_id: str, cohort: str) -> Optional[Dict]:
    """
    Get detailed patient information
    First tries patient_summary.csv, then falls back to assignments data
    
    Args:
        patient_id: Patient ID
        cohort: 'i10' or 'z01'
    
    Returns:
        Dictionary with patient details or None if not found
    """
    # First, try to load from patient_summary.csv
    patient_summary_path = Path("data/patient_summary.csv")
    
    if patient_summary_path.exists():
        try:
            patient_df = pd.read_csv(patient_summary_path)
            patient_row = patient_df[patient_df['pid'] == patient_id]
            
            if len(patient_row) > 0:
                patient_dict = patient_row.iloc[0].to_dict()
                # Ensure we have the key fields we need
                if patient_dict:
                    return patient_dict
        except Exception as e:
            # If loading fails, fall through to assignments fallback
            # Log the error for debugging (in production, use proper logging)
            import sys
            print(f"Warning: Could not load from patient_summary.csv for {patient_id}: {e}", file=sys.stderr)
    
    # Fallback: Use assignments data (which already has patient info)
    try:
        data = load_segmentation_data(cohort)
        assignments = data['assignments']
        patient_row = assignments[assignments['pid'] == patient_id]
        
        if len(patient_row) == 0:
            return None
        
        # Convert assignments row to dict and map columns to expected format
        patient_dict = patient_row.iloc[0].to_dict()
        
        # Map assignments columns to patient_details format
        # Assignments already has: sbp_latest, age, encounter_count_12m, icd3_count
        result = {}
        
        # Direct mappings (these should already be in assignments)
        if 'sbp_latest' in patient_dict:
            result['sbp_latest'] = patient_dict['sbp_latest']
        if 'age' in patient_dict:
            result['age'] = patient_dict['age']
        if 'encounter_count_12m' in patient_dict:
            result['encounter_count_12m'] = patient_dict['encounter_count_12m']
        if 'icd3_count' in patient_dict:
            result['icd3_count'] = patient_dict['icd3_count']
        if 'bmi_latest' in patient_dict:
            result['bmi_latest'] = patient_dict['bmi_latest']
        # Note: z01 has bmi_class, not bmi_latest in assignments
        
        # Include all other columns from assignments
        for key, value in patient_dict.items():
            if key not in result:
                result[key] = value
        
        return result if result else None
    except Exception as e:
        # Log the error for debugging
        import sys
        print(f"Error: Could not load patient details from assignments for {patient_id} in {cohort}: {e}", file=sys.stderr)
        return None


def compare_patient_to_cluster(patient_id: str, cohort: str, cluster_id: Optional[int] = None) -> Optional[Dict]:
    """
    Compare patient characteristics to their assigned cluster
    
    Args:
        patient_id: Patient ID
        cohort: 'i10' or 'z01'
        cluster_id: Optional cluster ID. If provided, skips lookup and uses this value.
                    If None, looks up cluster from assignments.
    
    Returns:
        Dictionary with comparison data or None if not found
    """
    # Get patient cluster (only if not provided)
    if cluster_id is None:
        cluster_id = get_patient_cluster(patient_id, cohort)
        if cluster_id is None:
            return None
    
    # Get patient details (will try patient_summary.csv first, then assignments)
    patient_details = get_patient_details(patient_id, cohort)
    
    if patient_details is None:
        return None
    
    # Get cluster summary
    cluster_summary = get_cluster_summary(cohort, cluster_id)
    
    if not cluster_summary:
        return None
    
    # Create comparison dictionary (always return this, even if no metrics match)
    comparison = {
        'patient_id': patient_id,
        'cluster_id': cluster_id,
        'cluster_name': cluster_summary.get('cluster_name', 'N/A'),
        'patient': {},
        'cluster': {},
        'difference': {}
    }
    
    # Key metrics to compare
    metrics_to_compare = []
    
    if 'sbp_latest' in patient_details and 'sbp_latest_median' in cluster_summary:
        metrics_to_compare.append(('sbp_latest', 'sbp_latest_median', 'SBP (mmHg)'))
    
    if 'age' in patient_details and 'age_median' in cluster_summary:
        metrics_to_compare.append(('age', 'age_median', 'Age (years)'))
    
    if 'bmi_latest' in patient_details and 'bmi_latest_median' in cluster_summary:
        metrics_to_compare.append(('bmi_latest', 'bmi_latest_median', 'BMI (kg/m²)'))
    
    if 'encounter_count_12m' in patient_details and 'encounter_count_12m_median' in cluster_summary:
        metrics_to_compare.append(('encounter_count_12m', 'encounter_count_12m_median', 'Encounters/Year'))
    
    if 'icd3_count' in patient_details and 'icd3_count_median' in cluster_summary:
        metrics_to_compare.append(('icd3_count', 'icd3_count_median', 'ICD-3 Count'))
    
    # Build comparison
    for patient_key, cluster_key, display_name in metrics_to_compare:
        patient_val = patient_details.get(patient_key)
        cluster_val = cluster_summary.get(cluster_key)
        
        # Handle both None and NaN values
        if patient_val is not None and cluster_val is not None:
            try:
                # Convert to numeric if possible, handle NaN
                patient_val_num = pd.to_numeric(patient_val, errors='coerce')
                cluster_val_num = pd.to_numeric(cluster_val, errors='coerce')
                
                if pd.notna(patient_val_num) and pd.notna(cluster_val_num):
                    comparison['patient'][display_name] = float(patient_val_num)
                    comparison['cluster'][display_name] = float(cluster_val_num)
                    comparison['difference'][display_name] = float(patient_val_num - cluster_val_num)
            except (ValueError, TypeError):
                # Skip if conversion fails
                pass
    
    # Always return comparison, even if no metrics were added
    # This allows the UI to at least show the cluster assignment
    return comparison

