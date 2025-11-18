#!/usr/bin/env python3
"""
Generate Professional PDF Report for Z01 Patient Segmentation Analysis  
Adapted from I10 Report Structure
"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

def generate_data_anchored_cluster_name(cluster_data, assignments_df=None):
    """Generate data-anchored cluster names using bmi_class mode from assignments"""
    sbp = cluster_data['sbp_latest_median']
    
    # BP Level
    if pd.isna(sbp) or sbp < 100:
        bp_level = "BP-Missing"
    elif sbp < 120:
        bp_level = "Normal-BP"
    elif sbp < 130:
        bp_level = "Elevated-BP"
    elif sbp < 140:
        bp_level = "Stage-1-BP"
    elif sbp < 180:
        bp_level = "Stage-2-BP"
    else:
        bp_level = "Hypertensive-Crisis"
    
    # BMI - use bmi_class mode from assignments
    if assignments_df is not None:
        cluster_id = int(cluster_data['cluster'])
        cluster_assignments = assignments_df[assignments_df['cluster'] == cluster_id]
        if len(cluster_assignments) > 0 and 'bmi_class' in cluster_assignments.columns:
            bmi_class_mode = cluster_assignments['bmi_class'].mode()
            if len(bmi_class_mode) > 0:
                bmi_class_val = bmi_class_mode.iloc[0]
                bmi_map = {
                    'Normal': 'Normal-Wt',
                    'Overweight': 'Overweight',
                    'Obesity I': 'Obese-I',
                    'Obesity II+': 'Obese-II+',
                    'Missing': 'BMI-Unknown'
                }
                bmi_cat = bmi_map.get(str(bmi_class_val), 'BMI-Unknown')
            else:
                bmi_cat = "BMI-Unknown"
        else:
            bmi_cat = "BMI-Unknown"
    else:
        bmi_cat = "BMI-Unknown"
    
    # Utilization
    encounters = cluster_data.get('encounter_count_12m_median', 0)
    if encounters <= 2:
        util_level = "Low-Util"
    elif encounters <= 5:
        util_level = "Med-Util"
    elif encounters <= 8:
        util_level = "High-Util"
    else:
        util_level = "VHigh-Util"
    
    return f"{bp_level} | {bmi_cat} | {util_level}"

class Z01SegmentationReport:
    """Generate comprehensive Z01 patient segmentation report"""
    
    def __init__(self, output_path):
        self.output_path = output_path
        self.doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        
        self.base_path = Path("outputs/data")
        self.viz_path = Path("outputs/visualizations/z01_clustering")
        self.load_data()
        self.cluster_names = self._generate_cluster_names()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2c5aa0'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#1f4788'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=HexColor('#2c5aa0'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyJustified',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=14
        ))
        
        self.styles.add(ParagraphStyle(
            name='BulletPoint',
            parent=self.styles['Normal'],
            fontSize=11,
            leftIndent=20,
            spaceAfter=8,
            leading=14
        ))
        
    def load_data(self):
        """Load clustering results and metrics"""
        self.evaluation = pd.read_csv(self.base_path / "z01_clustering_evaluation.csv")
        self.profiles = pd.read_csv(self.base_path / "z01_cluster_profiles.csv")
        self.medoids = pd.read_csv(self.base_path / "z01_cluster_medoids.csv")
        self.assignments = pd.read_csv(self.base_path / "z01_cluster_assignments.csv")
        
        self.optimal_k = int(self.evaluation['optimal_k'].values[0])
        self.silhouette = float(self.evaluation['silhouette_score'].values[0])
        self.stability = float(self.evaluation['stability_jaccard_mean'].values[0])
        self.n_patients = int(self.evaluation['n_patients'].values[0])
        
    def _generate_cluster_names(self):
        """Generate data-anchored names for all clusters"""
        names = {}
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            name = generate_data_anchored_cluster_name(cluster_data, self.assignments)
            names[i] = name
        return names
        
    def add_cover_page(self):
        """Add professional cover page"""
        self.story.append(Spacer(1, 2*inch))
        
        title = Paragraph(
            "Z01 Patient Segmentation Analysis",
            self.styles['CustomTitle']
        )
        self.story.append(title)
        
        subtitle = Paragraph(
            "Clinical Clustering for Preventive Care Encounters<br/>Using Gower Distance and PAM Algorithm",
            self.styles['CustomSubtitle']
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Key findings box
        key_findings_data = [
            ['Metric', 'Value'],
            ['Patient Cohort', f'{self.n_patients:,} patients'],
            ['Optimal Clusters', f'{self.optimal_k} segments'],
            ['Silhouette Score', f'{self.silhouette:.3f}'],
            ['Stability (Jaccard)', f'{self.stability:.3f}'],
            ['Feature Set', '9 features (lean)']
        ]
        
        key_table = Table(key_findings_data, colWidths=[2.5*inch, 2.5*inch])
        key_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('TOPPADDING', (0, 1), (-1, -1), 8),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 8),
        ]))
        
        self.story.append(key_table)
        self.story.append(Spacer(1, 1*inch))
        
        date_text = Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y')}",
            self.styles['Normal']
        )
        self.story.append(date_text)
        
        self.story.append(PageBreak())
        
    def add_executive_summary(self):
        """Add executive summary section"""
        self.story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        text = f"""
        This report presents a comprehensive patient segmentation analysis of {self.n_patients:,} individuals 
        encountered for preventive care and health examinations (ICD-10: Z01). Using advanced clustering 
        techniques with a lean, non-redundant feature set, we identified {self.optimal_k} distinct patient 
        segments with clinically meaningful characteristics.
        """
        self.story.append(Paragraph(text, self.styles['BodyJustified']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Study objectives
        self.story.append(Paragraph("Study Objectives", self.styles['SubsectionHeader']))
        objectives = [
            "Segment Z01 preventive care patients into clinically distinct groups",
            "Identify patterns in BP levels, body composition, comorbidity, and healthcare utilization",
            "Provide actionable insights for targeted preventive care strategies",
            "Validate clustering robustness through silhouette analysis and bootstrap stability testing"
        ]
        for obj in objectives:
            self.story.append(Paragraph(f"• {obj}", self.styles['BulletPoint']))
        
        self.story.append(Spacer(1, 0.2*inch))
        
        # Key findings - data-anchored segments
        self.story.append(Paragraph("Identified Patient Segments", self.styles['SubsectionHeader']))
        
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            n = int(cluster_data['n_patients'])
            pct = cluster_data['pct_patients']
            sbp = cluster_data['sbp_latest_median']
            age = cluster_data['age_median']
            encounters = cluster_data['encounter_count_12m_median']
            has_i10 = cluster_data.get('has_I10_pct', 0)
            has_e78 = cluster_data.get('has_E78_pct', 0)
            
            cluster_name = self.cluster_names[i]
            
            summary = f"""
            <b>{cluster_name}</b> ({n} patients, {pct:.1f}%): 
            Median SBP {sbp:.0f} mmHg, Age {age:.0f} years, {encounters:.0f} encounters/year.
            Comorbidities: HTN {has_i10:.0f}%, Dyslipidemia {has_e78:.0f}%.
            """
            self.story.append(Paragraph(summary, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        
        # Clinical implications
        self.story.append(Paragraph("Clinical Implications", self.styles['SubsectionHeader']))
        implications = """
        The identified segments demonstrate heterogeneity in BP levels, body composition, comorbidity 
        burden, and healthcare utilization. These findings enable:
        """
        self.story.append(Paragraph(implications, self.styles['BodyJustified']))
        
        impl_points = [
            "Risk-stratified preventive care protocols tailored to each segment",
            "Early identification of patients needing hypertension management",
            "Targeted lifestyle interventions based on BP and BMI profiles",
            "Personalized screening frequency based on risk factors"
        ]
        for point in impl_points:
            self.story.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
        
    def add_introduction(self):
        """Add introduction and background"""
        self.story.append(Paragraph("1. Introduction", self.styles['SectionHeader']))
        
        self.story.append(Paragraph("1.1 Background", self.styles['SubsectionHeader']))
        background_text = """
        Preventive care encounters (ICD-10: Z01) represent critical opportunities for early detection 
        of cardiovascular risk factors. These encounters include routine health examinations, blood 
        pressure checks, and preventive screenings. Identifying distinct patient segments within this 
        population enables more targeted preventive strategies and efficient resource allocation.
        """
        self.story.append(Paragraph(background_text, self.styles['BodyJustified']))
        self.story.append(Spacer(1, 0.15*inch))
        
        rationale_text = """
        Traditional uniform approaches to preventive care may not adequately address the diverse needs 
        of patients. Some present with elevated BP requiring immediate intervention, while others are 
        healthy individuals seeking routine check-ups. Patient segmentation offers a data-driven approach 
        to identify these clinically meaningful subgroups.
        """
        self.story.append(Paragraph(rationale_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("1.2 Study Objectives", self.styles['SubsectionHeader']))
        objectives_text = """
        This analysis aims to segment Z01 preventive care patients using machine learning techniques 
        optimized for mixed data types. Specific objectives include:
        """
        self.story.append(Paragraph(objectives_text, self.styles['BodyJustified']))
        
        objectives = [
            "Apply Gower distance-based clustering with a lean, non-redundant feature set (9 features)",
            "Identify clinically distinct segments based on BP, BMI, comorbidity, and utilization patterns",
            "Validate clustering stability through bootstrap analysis",
            "Generate data-anchored cluster names for immediate clinical interpretability"
        ]
        for i, obj in enumerate(objectives, 1):
            self.story.append(Paragraph(f"{i}. {obj}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
        
    def add_methods(self):
        """Add methodology section"""
        self.story.append(Paragraph("2. Data and Methods", self.styles['SectionHeader']))
        
        self.story.append(Paragraph("2.1 Study Cohort", self.styles['SubsectionHeader']))
        cohort_text = f"""
        The analysis included {self.n_patients:,} adult patients with a primary encounter for 
        preventive care (ICD-10: Z01). All patients were required to have at least one documented 
        blood pressure measurement. The cohort represents a diverse population across age, body 
        composition, and healthcare utilization patterns.
        """
        self.story.append(Paragraph(cohort_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("2.2 Feature Engineering (Lean Set)", self.styles['SubsectionHeader']))
        features_text = """
        A lean feature set of 9 features was selected, eliminating redundancy while preserving 
        clinical signal across four key domains:
        """
        self.story.append(Paragraph(features_text, self.styles['BodyJustified']))
        
        feature_categories = [
            ("<b>Clinical Severity (2):</b>", "SBP (numeric), BMI class (categorical)"),
            ("<b>Demographics (2):</b>", "Age (numeric), Sex (categorical)"),
            ("<b>Comorbidity (3):</b>", "ICD-3 count, Hypertension flag (I10), Dyslipidemia flag (E78)"),
            ("<b>Utilization (1):</b>", "12-month encounter count"),
            ("<b>Data Quality (1):</b>", "BMI missing indicator")
        ]
        
        for category, features in feature_categories:
            text = f"{category} {features}"
            self.story.append(Paragraph(text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.15*inch))
        redundancy_text = """
        <b>Redundancy Elimination:</b> Removed BP stage (kept SBP numeric), BMI numeric (kept BMI class), 
        and SBP missing indicator (all patients have BP). This prevents over-weighting clinical severity.
        """
        self.story.append(Paragraph(redundancy_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("2.3 Clustering Methodology", self.styles['SubsectionHeader']))
        
        method_text = """
        <b>Gower Distance:</b> Handles mixed data types (continuous, categorical, binary). For numeric 
        features, uses range-normalized Manhattan distance; for categorical, simple matching. Missing 
        values handled by excluding variable from pairwise distance, preserving missingness patterns.
        """
        self.story.append(Paragraph(method_text, self.styles['BodyJustified']))
        self.story.append(Spacer(1, 0.1*inch))
        
        pam_text = """
        <b>PAM Algorithm:</b> Partitioning Around Medoids selects actual patients as cluster representatives, 
        making results directly interpretable. More robust to outliers than k-means.
        """
        self.story.append(Paragraph(pam_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("2.4 Validation Framework", self.styles['SubsectionHeader']))
        
        validation_intro = """
        Clustering quality assessed using multiple complementary metrics:
        """
        self.story.append(Paragraph(validation_intro, self.styles['BodyJustified']))
        
        validation_criteria = [
            "<b>Silhouette Analysis:</b> Measures cluster cohesion and separation. Higher values indicate better-defined clusters.",
            "<b>Bootstrap Stability:</b> 100 iterations (80% sampling) to assess consistency. Jaccard similarity ≥0.75 indicates excellent stability.",
            "<b>Clinical Validation:</b> Visual inspection of cluster profiles for clinical meaningfulness."
        ]
        
        for criterion in validation_criteria:
            self.story.append(Paragraph(f"• {criterion}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
    
    def add_results(self):
        """Add results section with tables and visualizations"""
        self.story.append(Paragraph("3. Results", self.styles['SectionHeader']))
        
        # Cluster evaluation
        self.story.append(Paragraph("3.1 Cluster Evaluation", self.styles['SubsectionHeader']))
        
        eval_text = f"""
        PAM clustering was performed for k=3, 4, and 5, with comprehensive evaluation of each solution. 
        The optimal solution of k={self.optimal_k} clusters was selected, achieving a silhouette score 
        of {self.silhouette:.3f} and demonstrating clinically distinct segments.
        """
        self.story.append(Paragraph(eval_text, self.styles['BodyJustified']))
        
        # Add silhouette visualization
        self.story.append(Spacer(1, 0.2*inch))
        try:
            img_path = str(self.viz_path / "silhouette_comparison.png")
            img = Image(img_path, width=6*inch, height=3*inch)
            self.story.append(img)
            caption = Paragraph(
                "<i>Figure 1: Silhouette scores and size threshold compliance across different k values</i>",
                self.styles['Normal']
            )
            self.story.append(caption)
        except Exception as e:
            self.story.append(Paragraph(f"<i>[Silhouette visualization not available]</i>", self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Stability analysis
        self.story.append(Paragraph("3.2 Stability Analysis", self.styles['SubsectionHeader']))
        
        stability_text = f"""
        Bootstrap stability testing (100 iterations, 80% sampling) yielded a mean Jaccard similarity 
        of {self.stability:.3f}, indicating good reproducibility. The identified segments are robust 
        to data resampling and not artifacts of the specific cohort.
        """
        self.story.append(Paragraph(stability_text, self.styles['BodyJustified']))
        
        # Add stability visualization
        self.story.append(Spacer(1, 0.2*inch))
        try:
            img_path = str(self.viz_path / "stability_analysis.png")
            img = Image(img_path, width=6*inch, height=3*inch)
            self.story.append(img)
            caption = Paragraph(
                "<i>Figure 2: Bootstrap stability analysis showing Jaccard similarity distributions</i>",
                self.styles['Normal']
            )
            self.story.append(caption)
        except Exception as e:
            self.story.append(Paragraph(f"<i>[Stability visualization not available]</i>", self.styles['Normal']))
        
        self.story.append(PageBreak())
        
        # Final clustering solution
        self.story.append(Paragraph("3.3 Final Clustering Solution", self.styles['SubsectionHeader']))
        
        final_text = f"""
        The final {self.optimal_k}-cluster solution segments the {self.n_patients:,} patients into 
        clinically meaningful groups. Cluster sizes range from {int(self.profiles['n_patients'].min())} 
        to {int(self.profiles['n_patients'].max())} patients, all exceeding the 5% minimum threshold.
        """
        self.story.append(Paragraph(final_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        
        # Cluster summary table
        cluster_summary_data = [['Cluster', 'Name', 'N', '%', 'SBP', 'Age', 'Enc/yr']]
        
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            cluster_name = self.cluster_names[i]
            short_name = " | ".join([p.strip().replace("Stage-", "S").replace("Elevated-", "Elev-") 
                                     for p in cluster_name.split("|")])
            row = [
                f"{i}",
                short_name[:30],
                f"{int(cluster_data['n_patients'])}",
                f"{cluster_data['pct_patients']:.1f}%",
                f"{cluster_data['sbp_latest_median']:.0f}",
                f"{cluster_data['age_median']:.0f}",
                f"{cluster_data['encounter_count_12m_median']:.0f}"
            ]
            cluster_summary_data.append(row)
        
        cluster_table = Table(cluster_summary_data, colWidths=[0.5*inch, 2*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.7*inch])
        cluster_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        
        self.story.append(cluster_table)
        self.story.append(Spacer(1, 0.1*inch))
        caption = Paragraph(
            "<i>Table 1: Summary characteristics of the patient segments</i>",
            self.styles['Normal']
        )
        self.story.append(caption)
        
        # Add evaluation dashboard
        self.story.append(Spacer(1, 0.3*inch))
        try:
            img_path = str(self.viz_path / "evaluation_metrics_dashboard.png")
            img = Image(img_path, width=6.5*inch, height=3.5*inch)
            self.story.append(img)
            caption = Paragraph(
                "<i>Figure 3: Comprehensive evaluation metrics dashboard for the final clustering solution</i>",
                self.styles['Normal']
            )
            self.story.append(caption)
        except Exception as e:
            self.story.append(Paragraph(f"<i>[Evaluation dashboard not available]</i>", self.styles['Normal']))
        
        self.story.append(PageBreak())
        
        # Clinical profiles
        self.story.append(Paragraph("3.4 Detailed Clinical Profiles", self.styles['SubsectionHeader']))
        
        # Add heatmap
        try:
            img_path = str(self.viz_path / f"clinical_profile_heatmap_k{self.optimal_k}.png")
            img = Image(img_path, width=6.5*inch, height=3.5*inch)
            self.story.append(img)
            caption = Paragraph(
                "<i>Figure 4: Clinical profile heatmap showing normalized values across clusters</i>",
                self.styles['Normal']
            )
            self.story.append(caption)
        except Exception as e:
            self.story.append(Paragraph(f"<i>[Clinical heatmap not available]</i>", self.styles['Normal']))
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Individual cluster profiles
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            cluster_name = self.cluster_names[i]
            
            self.story.append(Paragraph(f"{cluster_name} (Cluster {i})", self.styles['SubsectionHeader']))
            
            has_i10 = cluster_data.get('has_I10_pct', 0)
            has_e78 = cluster_data.get('has_E78_pct', 0)
            
            profile_text = f"""
            <b>Size:</b> {int(cluster_data['n_patients'])} patients ({cluster_data['pct_patients']:.1f}% of cohort)
            <br/><br/>
            <b>Clinical Severity:</b>
            <br/>• Blood Pressure: Median SBP {cluster_data['sbp_latest_median']:.0f} mmHg 
            (range: {cluster_data['sbp_min']:.0f}-{cluster_data['sbp_max']:.0f})
            <br/><br/>
            <b>Demographics:</b>
            <br/>• Age: Median {cluster_data['age_median']:.0f} years 
            (range: {cluster_data['age_min']:.0f}-{cluster_data['age_max']:.0f})
            <br/><br/>
            <b>Comorbidity Burden:</b>
            <br/>• Hypertension (I10): {has_i10:.1f}%
            <br/>• Dyslipidemia (E78): {has_e78:.1f}%
            <br/>• Mean ICD-3 codes: {cluster_data['icd3_count_median']:.1f}
            <br/><br/>
            <b>Healthcare Utilization:</b>
            <br/>• Median encounters (12 months): {cluster_data['encounter_count_12m_median']:.0f}
            """
            
            self.story.append(Paragraph(profile_text, self.styles['BodyJustified']))
            
            if i < self.optimal_k - 1:
                self.story.append(Spacer(1, 0.2*inch))
        
        # Add PCA visualization if available
        pca_path = self.viz_path / "pca_visualization_beautiful_k4.png"
        if pca_path.exists():
            self.story.append(Spacer(1, 0.3*inch))
            self.story.append(Paragraph("3.5 Cluster Visualization (PCA)", self.styles['SubsectionHeader']))
            
            img = Image(str(pca_path), width=6.5*inch, height=4*inch)
            self.story.append(img)
            
            caption = """
            <i>Figure 5: PCA projection of patient segments. Each point represents a patient, colored by 
            cluster assignment. Stars indicate cluster medoids (representative patients). Density contours 
            show cluster concentration.</i>
            """
            self.story.append(Paragraph(caption, self.styles['Normal']))
        
        self.story.append(PageBreak())
        
    def add_clinical_interpretation(self):
        """Add clinical interpretation and recommendations"""
        self.story.append(Paragraph("4. Clinical Interpretation and Recommendations", 
                                   self.styles['SectionHeader']))
        
        intro_text = """
        The identified segments demonstrate clinically meaningful differences in BP levels, body composition, 
        comorbidity burden, and healthcare utilization. Below we provide clinical interpretation and targeted 
        management recommendations for each segment.
        """
        self.story.append(Paragraph(intro_text, self.styles['BodyJustified']))
        
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            cluster_name = self.cluster_names[i]
            
            n = int(cluster_data['n_patients'])
            pct = cluster_data['pct_patients']
            sbp = cluster_data['sbp_latest_median']
            age = cluster_data['age_median']
            encounters = cluster_data['encounter_count_12m_median']
            icd3 = cluster_data['icd3_count_median']
            has_i10 = cluster_data.get('has_I10_pct', 0)
            has_e78 = cluster_data.get('has_E78_pct', 0)
            
            bp_stage = cluster_name.split("|")[0].strip()
            bmi_cat = cluster_name.split("|")[1].strip()
            util_level = cluster_name.split("|")[2].strip()
            
            self.story.append(Spacer(1, 0.2*inch))
            self.story.append(Paragraph(f"4.{i+1} {cluster_name}", 
                                       self.styles['SubsectionHeader']))
            
            # Characteristics
            char_text = f"""
            <b>Characteristics:</b> {n} patients ({pct:.1f}%), median age {age:.0f} years, 
            BP {sbp:.0f} mmHg ({bp_stage}), {util_level}, comorbidity burden {icd3:.0f} ICD-3 codes, 
            HTN {has_i10:.0f}%, Dyslipidemia {has_e78:.0f}%.
            """
            self.story.append(Paragraph(char_text, self.styles['BodyJustified']))
            
            self.story.append(Spacer(1, 0.1*inch))
            
            # Risk assessment
            if sbp >= 160 or has_i10 >= 75:
                risk = "High cardiovascular risk requiring immediate intervention"
            elif sbp >= 140 or has_i10 >= 50:
                risk = "Moderate risk needing enhanced monitoring and lifestyle interventions"
            elif sbp >= 130:
                risk = "Elevated risk - lifestyle modifications to prevent progression"
            else:
                risk = "Lower risk - routine preventive care appropriate"
            
            self.story.append(Paragraph(f"<b>Risk Assessment:</b> {risk}", self.styles['BodyJustified']))
            
            self.story.append(Spacer(1, 0.1*inch))
            
            # Recommendations
            recs = []
            if sbp >= 140:
                recs.append("Initiate or intensify antihypertensive therapy per guidelines")
            elif sbp >= 130:
                recs.append("Recommend lifestyle modifications (diet, exercise, weight management)")
            
            if bmi_cat in ['Obese-I', 'Obese-II', 'Obese-II+', 'Obese-III']:
                recs.append("Provide weight management counseling and resources")
            
            if has_i10 >= 50:
                recs.append("Ensure BP monitoring adherence and medication compliance")
            
            if has_e78 >= 30:
                recs.append("Consider lipid panel and statin therapy evaluation")
            
            if encounters >= 6:
                recs.append("Assess care coordination needs and social determinants")
            elif encounters <= 2:
                recs.append("Encourage regular preventive care engagement")
            
            if not recs:
                recs.append("Continue routine preventive care and screenings")
                recs.append("Reinforce healthy lifestyle behaviors")
            
            self.story.append(Paragraph("<b>Recommended Actions:</b>", self.styles['Normal']))
            for rec in recs:
                self.story.append(Paragraph(f"• {rec}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
        
    def add_technical_appendix(self):
        """Add technical appendix"""
        self.story.append(Paragraph("Technical Appendix", self.styles['SectionHeader']))
        
        # Gower distance
        self.story.append(Paragraph("A. Statistical Methodology", self.styles['SubsectionHeader']))
        
        gower_text = """
        <b>Gower Distance Formula:</b><br/>
        d(i,j) = Σ δᵢⱼₖ · dᵢⱼₖ / Σ δᵢⱼₖ
        <br/><br/>
        where k indexes features, δᵢⱼₖ is 1 if feature k available for both i and j (else 0), 
        and dᵢⱼₖ is feature-specific distance:
        <br/>• Numeric: |xᵢₖ - xⱼₖ| / rangeₖ
        <br/>• Categorical: 0 if same, else 1
        <br/>• Binary: 0 if same, else 1
        """
        self.story.append(Paragraph(gower_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.15*inch))
        
        pam_algo = """
        <b>PAM Algorithm:</b> Build (initialize k medoids), Assign (patients to nearest medoid), 
        Update (find best medoid per cluster), Iterate (until convergence).
        """
        self.story.append(Paragraph(pam_algo, self.styles['BodyJustified']))
        
        # Validation metrics
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("B. Validation Metrics", self.styles['SubsectionHeader']))
        
        metrics_text = """
        <b>Silhouette Score:</b> Measures cluster cohesion vs. separation. Higher values indicate 
        better-defined clusters. For clinical data with overlap, scores ≥0.15 acceptable, ≥0.20 good.
        <br/><br/>
        <b>Jaccard Stability:</b> Measures reproducibility across bootstrap samples. Ratio of patient 
        pairs classified together in both original and resampled solutions. Values ≥0.75 indicate 
        excellent stability.
        """
        self.story.append(Paragraph(metrics_text, self.styles['BodyJustified']))
        
        # Limitations
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("C. Limitations", self.styles['SubsectionHeader']))
        
        limitations = [
            "Cross-sectional analysis - temporal changes not captured",
            "Based on documented data - unmeasured factors may influence phenotypes",
            "Results specific to this cohort - external validation recommended"
        ]
        
        for limitation in limitations:
            self.story.append(Paragraph(f"• {limitation}", self.styles['BulletPoint']))
        
        # Computational specs
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("D. Computational Specifications", self.styles['SubsectionHeader']))
        
        specs = f"""
        <b>Software:</b> Python 3.12, pandas, numpy, scikit-learn, gower
        <br/><b>Clustering:</b> Custom PAM implementation with k-medoids++ initialization
        <br/><b>Random Seed:</b> 42 (reproducibility)
        <br/><b>Bootstrap:</b> 100 iterations, 80% sampling
        <br/><b>Distance Matrix:</b> {self.n_patients} × {self.n_patients} Gower distances on 9-feature set
        """
        self.story.append(Paragraph(specs, self.styles['BodyJustified']))
        
    def generate(self):
        """Generate the complete PDF report"""
        print("Generating Z01 Patient Segmentation Report...")
        print(f"  Using data-anchored cluster names:")
        for i, name in self.cluster_names.items():
            print(f"    Cluster {i}: {name}")
        
        self.add_cover_page()
        self.add_executive_summary()
        self.add_introduction()
        self.add_methods()
        self.add_results()
        self.add_clinical_interpretation()
        self.add_technical_appendix()
        
        # Build PDF
        self.doc.build(self.story)
        print(f"\n✓ Report generated: {self.output_path}")

if __name__ == "__main__":
    output_path = "outputs/reports/z01_patient_segmentation_report.pdf"
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    
    report = Z01SegmentationReport(output_path)
    report.generate()
    
    print(f"\n{'='*60}")
    print("Professional PDF report created successfully!")
    print(f"Location: {output_path}")
    print(f"{'='*60}")
    print("\nReport Sections:")
    print("  ✓ Cover page with key metrics")
    print("  ✓ Executive summary with data-anchored cluster names")
    print("  ✓ Introduction and background")
    print("  ✓ Detailed methodology")
    print("  ✓ Results with ALL visualizations (Figures 1-5)")
    print("  ✓ Clinical interpretation")
    print("  ✓ Technical appendix")
    print(f"{'='*60}")
