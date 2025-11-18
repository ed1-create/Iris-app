#!/usr/bin/env python3
"""
Generate REFINED Professional PDF Report for I10 Patient Segmentation Analysis
WITH DATA-ANCHORED CLUSTER NAMES
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

def generate_data_anchored_cluster_name(cluster_data):
    """
    Generate precise, data-anchored cluster name based on actual characteristics.
    Format: "[BP Level] | [BMI Category] | [Utilization Level]"
    """
    # Get median SBP
    sbp = cluster_data['sbp_latest_median']
    
    # BP Level classification
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
    
    # BMI Category - try to get from bmi_class mode or calculate from bmi_latest
    bmi = cluster_data.get('bmi_latest_median', None)
    if pd.notna(bmi):
        if bmi < 18.5:
            bmi_cat = "Underweight"
        elif bmi < 25:
            bmi_cat = "Normal-Wt"
        elif bmi < 30:
            bmi_cat = "Overweight"
        elif bmi < 35:
            bmi_cat = "Obese-I"
        elif bmi < 40:
            bmi_cat = "Obese-II"
        else:
            bmi_cat = "Obese-III"
    else:
        bmi_cat = "BMI-Unknown"
    
    # Utilization Level
    encounters = cluster_data.get('encounter_count_12m_median', 0)
    if encounters <= 2:
        util_level = "Low-Util"
    elif encounters <= 5:
        util_level = "Med-Util"
    elif encounters <= 8:
        util_level = "High-Util"
    else:
        util_level = "VHigh-Util"
    
    # Construct name
    name = f"{bp_level} | {bmi_cat} | {util_level}"
    
    return name

def generate_clinical_description(cluster_data, cluster_name):
    """Generate clinical description based on cluster characteristics"""
    n = int(cluster_data['n_patients'])
    pct = cluster_data['pct_patients']
    sbp = cluster_data['sbp_latest_median']
    age = cluster_data['age_median']
    bmi = cluster_data.get('bmi_latest_median', cluster_data.get('bmi_class', 'Unknown'))
    encounters = cluster_data['encounter_count_12m_median']
    
    # Risk assessment
    if sbp >= 160 or encounters >= 8:
        risk = "High"
    elif sbp >= 140 or encounters >= 5:
        risk = "Moderate"
    else:
        risk = "Lower"
    
    description = f"""
    <b>{cluster_name}</b>
    <br/><br/>
    <b>Cohort:</b> {n} patients ({pct:.1f}%)
    <br/><b>Clinical Profile:</b> Median SBP {sbp:.0f} mmHg, Age {age:.0f} years, {encounters:.0f} encounters/year
    <br/><b>Risk Level:</b> {risk} cardiovascular risk
    """
    
    return description

class RefinedI10SegmentationReport:
    """Generate comprehensive I10 patient segmentation report with data-anchored names"""
    
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
        
        # Load data
        self.base_path = Path("outputs/data")
        self.viz_path = Path("outputs/visualizations/i10_clustering")
        self.load_data()
        
        # Generate data-anchored names
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
        self.evaluation = pd.read_csv(self.base_path / "i10_clustering_evaluation.csv")
        self.profiles = pd.read_csv(self.base_path / "i10_cluster_profiles.csv")
        self.medoids = pd.read_csv(self.base_path / "i10_cluster_medoids.csv")
        
        self.optimal_k = int(self.evaluation['optimal_k'].values[0])
        self.silhouette = float(self.evaluation['silhouette_score'].values[0])
        self.stability = float(self.evaluation['stability_jaccard_mean'].values[0])
        self.n_patients = int(self.evaluation['n_patients'].values[0])
        
    def _generate_cluster_names(self):
        """Generate data-anchored names for all clusters"""
        names = {}
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            name = generate_data_anchored_cluster_name(cluster_data)
            names[i] = name
        return names
        
    def add_cover_page(self):
        """Add professional cover page with improvements note"""
        self.story.append(Spacer(1, 1.5*inch))
        
        title = Paragraph(
            "I10 Patient Segmentation Analysis<br/><font size=18>(REFINED VERSION)</font>",
            self.styles['CustomTitle']
        )
        self.story.append(title)
        
        subtitle = Paragraph(
            "Clinical Clustering with Data-Anchored Segment Names<br/>Gower Distance + PAM Algorithm",
            self.styles['CustomSubtitle']
        )
        self.story.append(subtitle)
        self.story.append(Spacer(1, 0.3*inch))
        
        # Improvements box
        improvements_text = Paragraph(
            "<b>KEY IMPROVEMENTS IN THIS VERSION:</b>",
            self.styles['Normal']
        )
        self.story.append(improvements_text)
        self.story.append(Spacer(1, 0.1*inch))
        
        improvements = [
            "✓ Reduced feature redundancy (12 vs 16 features)",
            "✓ Proper missing data handling (preserved care gaps)",
            "✓ Data-anchored cluster names (BP | BMI | Utilization)",
            "✓ Focused k range (3-5 vs 3-7)",
            "✓ Improved stability targeting"
        ]
        
        for imp in improvements:
            self.story.append(Paragraph(imp, self.styles['BulletPoint']))
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Key findings
        key_findings_data = [
            ['Metric', 'Value'],
            ['Patient Cohort', f'{self.n_patients:,} patients'],
            ['Optimal Clusters', f'{self.optimal_k} segments'],
            ['Silhouette Score', f'{self.silhouette:.3f}'],
            ['Stability (Jaccard)', f'{self.stability:.3f}'],
            ['Feature Set', '12 features (refined)']
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
        self.story.append(Spacer(1, 0.5*inch))
        
        date_text = Paragraph(
            f"<b>Report Generated:</b> {datetime.now().strftime('%B %d, %Y')}",
            self.styles['Normal']
        )
        self.story.append(date_text)
        
        self.story.append(PageBreak())
        
    def add_executive_summary(self):
        """Add executive summary with data-anchored segment names"""
        self.story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
        
        text = f"""
        This REFINED analysis presents patient segmentation of {self.n_patients:,} individuals 
        with essential hypertension (ICD-10: I10). Key improvements include removing feature redundancy, 
        properly handling missing data, and generating data-anchored cluster names for clinical clarity.
        The analysis identified {self.optimal_k} distinct patient segments.
        """
        self.story.append(Paragraph(text, self.styles['BodyJustified']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Identified segments with data-anchored names
        self.story.append(Paragraph("Identified Patient Segments", self.styles['SubsectionHeader']))
        
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            cluster_name = self.cluster_names[i]
            
            desc = generate_clinical_description(cluster_data, cluster_name)
            self.story.append(Paragraph(desc, self.styles['BodyJustified']))
            self.story.append(Spacer(1, 0.15*inch))
        
        self.story.append(PageBreak())
        
    def add_refinement_rationale(self):
        """Add section explaining the refinements made"""
        self.story.append(Paragraph("Refinement Rationale", self.styles['SectionHeader']))
        
        rationale_text = """
        This refined analysis addresses critical issues identified in the initial segmentation:
        """
        self.story.append(Paragraph(rationale_text, self.styles['BodyJustified']))
        
        issues_addressed = [
            ("<b>Feature Redundancy:</b>", 
             "Removed double-counting of BP (kept SBP numeric only), BMI (kept class only), and age (kept numeric only). This prevents over-weighting clinical severity relative to utilization and comorbidity."),
            ("<b>Missing Data Handling:</b>", 
             "Preserved missing SBP values as they indicate care gaps and workflow differences. Previous median imputation artificially removed this valuable clinical signal."),
            ("<b>Model Parsimony:</b>", 
             "Reduced from 16 to 12 features and focused k range on 3-5 (vs 3-7) to improve stability and interpretability."),
            ("<b>Cluster Naming:</b>", 
             "Generated data-anchored names in format '[BP Level] | [BMI Category] | [Utilization Level]' instead of generic descriptive names."),
        ]
        
        for title, desc in issues_addressed:
            combined = f"{title} {desc}"
            self.story.append(Paragraph(combined, self.styles['BodyJustified']))
        
        self.story.append(PageBreak())
    
    def generate(self):
        """Generate the complete PDF report"""
        print("Generating REFINED I10 Patient Segmentation Report...")
        print(f"  Using data-anchored cluster names:")
        for i, name in self.cluster_names.items():
            print(f"    Cluster {i}: {name}")
        
        self.add_cover_page()
        self.add_executive_summary()
        self.add_refinement_rationale()
        
        # Note: Add remaining sections similar to original report
        # but with updated cluster names throughout
        
        # Build PDF
        self.doc.build(self.story)
        print(f"\n✓ Refined report generated: {self.output_path}")

if __name__ == "__main__":
    output_path = "outputs/reports/i10_patient_segmentation_report_refined.pdf"
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    
    report = RefinedI10SegmentationReport(output_path)
    report.generate()
    
    print(f"\n{'='*60}")
    print("REFINED Professional PDF report created successfully!")
    print(f"Location: {output_path}")
    print(f"{'='*60}")
    print("\nKey Improvements:")
    print("  ✓ Data-anchored cluster names")
    print("  ✓ Refinement rationale documented")
    print("  ✓ Improved feature set (12 features)")
    print("  ✓ Proper missing data handling")
    print(f"{'='*60}")

