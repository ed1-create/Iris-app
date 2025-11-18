"""
Generate Comprehensive EDA Report PDF
This script creates a detailed technical data science report documenting all
exploratory data analysis performed on the patient records dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from PIL import Image as PILImage
import os


# Configuration
OUTPUT_DIR = Path('outputs')
VIZ_DIR = OUTPUT_DIR / 'visualizations'
DATA_DIR = OUTPUT_DIR / 'data'
REPORTS_DIR = OUTPUT_DIR / 'reports'
REPORTS_DIR.mkdir(exist_ok=True)

# Load the main dataset for statistics
df = pd.read_csv('../data/patient_records.csv', low_memory=False)

# Convert date columns
date_columns = ['created', 'birth_date', 'ultrasound3_date']
for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Calculate age if available
if 'birth_date' in df.columns and 'created' in df.columns:
    df['age'] = (df['created'] - df['birth_date']).dt.days / 365.25
    df['age'] = df['age'].round(1)


class PDFReportGenerator:
    def __init__(self, output_path):
        self.output_path = output_path
        self.doc = SimpleDocTemplate(
            str(output_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        self.story = []
        self.styles = getSampleStyleSheet()
        
        # Custom styles
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1a1a1a'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        )
        
        self.heading1_style = ParagraphStyle(
            'CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=HexColor('#2c3e50'),
            spaceAfter=12,
            spaceBefore=12,
            fontName='Helvetica-Bold'
        )
        
        self.heading2_style = ParagraphStyle(
            'CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=HexColor('#34495e'),
            spaceAfter=10,
            spaceBefore=10,
            fontName='Helvetica-Bold'
        )
        
        self.body_style = ParagraphStyle(
            'CustomBody',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=HexColor('#333333'),
            spaceAfter=6,
            alignment=TA_JUSTIFY,
            leading=14
        )
        
        self.bullet_style = ParagraphStyle(
            'CustomBullet',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=HexColor('#333333'),
            spaceAfter=4,
            leftIndent=20,
            bulletIndent=10
        )
    
    def add_title_page(self):
        """Add title page"""
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph("Patient Segmentation Project", self.title_style))
        self.story.append(Spacer(1, 0.3*inch))
        self.story.append(Paragraph("Initial Data Exploration Report", self.title_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        date_style = ParagraphStyle(
            'DateStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            alignment=TA_CENTER,
            textColor=HexColor('#666666')
        )
        self.story.append(Paragraph(f"Generated on: {datetime.now().strftime('%B %d, %Y')}", date_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        info_style = ParagraphStyle(
            'InfoStyle',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_CENTER,
            textColor=HexColor('#555555')
        )
        self.story.append(Paragraph(f"Dataset: {len(df):,} records, {len(df.columns)} columns", info_style))
        self.story.append(Paragraph(f"Unique Patients: {df['pid'].nunique():,}", info_style))
        if 'created' in df.columns:
            date_range = f"{df['created'].min().strftime('%Y-%m-%d')} to {df['created'].max().strftime('%Y-%m-%d')}"
            self.story.append(Paragraph(f"Date Range: {date_range}", info_style))
        
        self.story.append(PageBreak())
    
    def add_image(self, image_path, width=6*inch, caption=None):
        """Add image with optional caption"""
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        
        if image_path.exists():
            # Resize image if needed
            try:
                img = PILImage.open(image_path)
                img_width, img_height = img.size
                aspect_ratio = img_height / img_width
                height = width * aspect_ratio
                
                # Limit height to prevent page overflow
                if height > 7*inch:
                    height = 7*inch
                    width = height / aspect_ratio
                
                self.story.append(Image(str(image_path), width=width, height=height))
                if caption:
                    caption_style = ParagraphStyle(
                        'CaptionStyle',
                        parent=self.styles['Normal'],
                        fontSize=9,
                        alignment=TA_CENTER,
                        textColor=HexColor('#666666'),
                        spaceAfter=12
                    )
                    self.story.append(Paragraph(caption, caption_style))
                self.story.append(Spacer(1, 0.2*inch))
                return True
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                return False
        else:
            print(f"Image not found: {image_path}")
            return False
    
    def add_table(self, data, headers=None, col_widths=None, header_font_size=10, body_font_size=9):
        """Add table to report
        
        Args:
            data: DataFrame or list of lists
            headers: List of header names (if not provided, uses DataFrame columns)
            col_widths: List of column widths in inches
            header_font_size: Font size for header row (default: 10)
            body_font_size: Font size for body rows (default: 9)
        """
        if isinstance(data, pd.DataFrame):
            if headers is None:
                headers = list(data.columns)
            data_list = [headers] + data.values.tolist()
        else:
            data_list = data
        
        if col_widths is None:
            col_widths = [1.5*inch] * len(headers) if headers else [1.5*inch] * len(data_list[0])
        
        table = Table(data_list, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#34495e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), header_font_size),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), HexColor('#ecf0f1')),
            ('TEXTCOLOR', (0, 1), (-1, -1), HexColor('#2c3e50')),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), body_font_size),
            ('GRID', (0, 0), (-1, -1), 1, HexColor('#bdc3c7')),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, HexColor('#f8f9fa')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_executive_summary(self):
        """Add executive summary section"""
        self.story.append(Paragraph("Executive Summary", self.heading1_style))
        
        # Key statistics
        total_records = len(df)
        unique_patients = df['pid'].nunique()
        total_columns = len(df.columns)
        
        # Calculate data quality score
        total_cells = total_records * total_columns
        missing_cells = df.isnull().sum().sum()
        quality_score = (1 - (missing_cells / total_cells)) * 100
        
        summary_text = f"""
        This report presents a comprehensive exploratory data analysis of the patient records dataset 
        containing {total_records:,} medical encounters from {unique_patients:,} unique patients. 
        The dataset comprises {total_columns} columns covering clinical measurements, demographics, 
        diagnostic codes, and healthcare utilization patterns.
        
        <b>Key Findings:</b>
        • Dataset contains {total_records:,} records across {total_columns} variables
        • {unique_patients:,} unique patients identified through hashed patient identifiers
        • Overall data quality score: {quality_score:.2f}%
        • Missing data patterns vary significantly by clinical variable type
        • Anthropometric measurements show age-dependent completeness patterns
        • Blood pressure data available for approximately 31% of records
        • Three primary ICD3 codes analyzed: E07 (Thyroid), I10 (Hypertension), E11 (Diabetes)
        """
        
        self.story.append(Paragraph(summary_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Key statistics table
        stats_data = [
            ['Metric', 'Value'],
            ['Total Records', f"{total_records:,}"],
            ['Unique Patients', f"{unique_patients:,}"],
            ['Total Columns', str(total_columns)],
            ['Data Quality Score', f"{quality_score:.2f}%"],
            ['Duplicate Records', str(df.duplicated().sum())],
        ]
        
        if 'created' in df.columns:
            stats_data.append(['Date Range', f"{df['created'].min().strftime('%Y-%m-%d')} to {df['created'].max().strftime('%Y-%m-%d')}"])
        
        self.add_table(stats_data, col_widths=[3*inch, 2.5*inch])
        self.story.append(PageBreak())
    
    def add_dataset_overview(self):
        """Add dataset overview section"""
        self.story.append(Paragraph("Dataset Overview", self.heading1_style))
        
        overview_text = """
        The patient records dataset was processed from the original healthcare_translated.csv file. 
        Patient identifiers (taj_identifier) were hashed using SHA256 with a salt key to create 
        anonymized patient IDs (pid) for privacy compliance. The dataset includes comprehensive 
        clinical measurements, demographic information, diagnostic codes, and healthcare utilization data.
        """
        self.story.append(Paragraph(overview_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Column categories
        self.story.append(Paragraph("Column Categories", self.heading2_style))
        
        # Categorize columns
        column_categories = {
            'Demographics': ['pid', 'patient_name', 'birth_date', 'patient_gender', 'age'],
            'Location': ['mep', 'mep_region', 'settlement', 'clinic_name'],
            'Clinical Measurements': ['bp_systolic', 'bp_diastolic', 'pulse', 'physical1_height', 
                                     'physical2_weight', 'physical3_bmi', 'cv_screening5_height',
                                     'cv_screening6_weight', 'cv_screening7_bmi'],
            'Diagnostics': ['icd3_code', 'icd_code_name', 'specialty_name'],
            'Ultrasound': ['ultrasound_description', 'ultrasound1_area_code_id', 'ultrasound3_date'],
            'Medications': ['prescribed_medication', 'prescribed_medication_atc'],
            'Screening': [col for col in df.columns if 'cv_screening' in col or 'screening' in col],
            'Other': []
        }
        
        # Find uncategorized columns
        all_categorized = set()
        for cat_cols in column_categories.values():
            all_categorized.update(cat_cols)
        
        column_categories['Other'] = [col for col in df.columns if col not in all_categorized]
        
        cat_text = ""
        for category, cols in column_categories.items():
            if cols:
                available_cols = [c for c in cols if c in df.columns]
                if available_cols:
                    cat_text += f"<b>{category}:</b> {len(available_cols)} columns<br/>"
        
        self.story.append(Paragraph(cat_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Key column explanations
        self.story.append(Paragraph("Key Column Explanations", self.heading2_style))
        
        explanations = [
            ("mep", "Medical examination point identifier - represents the healthcare facility location"),
            ("specialty_name", "Medical specialty associated with the encounter (e.g., endocrinology, cardiology)"),
            ("pid", "Patient identifier - SHA256 hash of taj_identifier with salt key for privacy"),
            ("taj_present", "Indicates whether original taj_identifier was available (yes/no)"),
            ("icd3_code", "First 3 characters of ICD-10 diagnostic code"),
            ("cv_screening", "Cardiovascular screening measurements (height, weight, BMI, BP)"),
            ("physical", "Physical examination measurements (height, weight, BMI, waist)")
        ]
        
        expl_text = ""
        for term, desc in explanations:
            expl_text += f"<b>{term}:</b> {desc}<br/><br/>"
        
        self.story.append(Paragraph(expl_text, self.body_style))
        self.story.append(PageBreak())
    
    def add_data_quality(self):
        """Add data quality assessment section"""
        self.story.append(Paragraph("Data Quality Assessment", self.heading1_style))
        
        # Missing values analysis
        self.story.append(Paragraph("Missing Values Analysis", self.heading2_style))
        
        missing_counts = df.isnull().sum().sort_values(ascending=False)
        missing_pct = (missing_counts / len(df) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Column': missing_counts.index,
            'Missing Count': missing_counts.values,
            'Missing Percentage': missing_pct.values
        }).head(20)
        
        self.add_table(missing_df, col_widths=[3*inch, 1.5*inch, 1.5*inch])
        
        # Add missing data heatmap if available
        heatmap_path = VIZ_DIR / 'data_quality' / 'missing_data_heatmap.png'
        if heatmap_path.exists():
            self.story.append(Paragraph("Missing Data Pattern Heatmap", self.heading2_style))
            self.add_image(heatmap_path, width=6*inch, caption="Heatmap showing missing data patterns across columns")
        
        # Outlier detection
        self.story.append(Paragraph("Outlier Detection", self.heading2_style))
        
        outlier_text = """
        Outlier detection was performed on key clinical variables using the Interquartile Range (IQR) method. 
        Values beyond 1.5 * IQR from Q1 and Q3 were flagged as outliers. This analysis helps identify 
        potential data entry errors or unusual clinical values that may require special handling.
        """
        self.story.append(Paragraph(outlier_text, self.body_style))
        
        # Add outlier visualization
        outlier_path = VIZ_DIR / 'data_quality' / 'outlier_detection.png'
        if outlier_path.exists():
            self.add_image(outlier_path, width=6*inch, caption="Outlier detection for key clinical variables")
        
        # Data quality score
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        quality_score = (1 - (missing_cells / total_cells)) * 100
        
        quality_text = f"""
        <b>Overall Data Quality Score: {quality_score:.2f}%</b><br/><br/>
        This score represents the percentage of non-missing cells across the entire dataset. 
        While many columns have high missing rates, the core patient identifiers and encounter 
        information are consistently available.
        """
        self.story.append(Paragraph(quality_text, self.body_style))
        self.story.append(PageBreak())
    
    def add_anthropometric_analysis(self):
        """Add anthropometric data analysis section"""
        self.story.append(Paragraph("Anthropometric Data Analysis", self.heading1_style))
        
        analysis_text = """
        Anthropometric measurements (height, weight, BMI) are critical for patient segmentation. 
        The dataset contains measurements from two sources: cardiovascular screening (cv_screening) 
        and physical examinations (physical). This section analyzes data completeness patterns, 
        particularly how missing data relates to patient age.
        """
        self.story.append(Paragraph(analysis_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Add visualizations
        missing_age_path = VIZ_DIR / 'data_quality' / 'anthropometric_missing_by_age.png'
        if missing_age_path.exists():
            self.story.append(Paragraph("Missing Data Patterns by Age Group", self.heading2_style))
            self.add_image(missing_age_path, width=6*inch, 
                          caption="Percentage of missing anthropometric measurements by age group")
        
        cv_vs_physical_path = VIZ_DIR / 'data_quality' / 'cv_vs_physical_comparison.png'
        if cv_vs_physical_path.exists():
            self.story.append(Paragraph("CV Screening vs Physical Measurements", self.heading2_style))
            self.add_image(cv_vs_physical_path, width=6*inch,
                          caption="Comparison of data availability between CV screening and physical examination measurements")
        
        age_anthro_path = VIZ_DIR / 'data_quality' / 'age_anthropometric_relationship.png'
        if age_anthro_path.exists():
            self.story.append(Paragraph("Age and Anthropometric Data Relationships", self.heading2_style))
            self.add_image(age_anthro_path, width=6*inch,
                          caption="Relationship between age and anthropometric measurements")
        
        self.story.append(PageBreak())
    
    def add_demographics(self):
        """Add patient demographics section"""
        self.story.append(Paragraph("Patient Demographics", self.heading1_style))
        
        # Gender analysis
        if Path(DATA_DIR / 'patient_gender_unique.csv').exists():
            gender_df = pd.read_csv(DATA_DIR / 'patient_gender_unique.csv')
            
            self.story.append(Paragraph("Patient-Level Gender Distribution", self.heading2_style))
            
            gender_counts = gender_df['gender'].value_counts()
            gender_table = pd.DataFrame({
                'Gender': gender_counts.index,
                'Count': gender_counts.values,
                'Percentage': (gender_counts.values / len(gender_df) * 100).round(2)
            })
            
            self.add_table(gender_table, col_widths=[2*inch, 2*inch, 2*inch])
            
            gender_viz_path = VIZ_DIR / 'data_quality' / 'patient_gender_analysis.png'
            if gender_viz_path.exists():
                self.add_image(gender_viz_path, width=6*inch,
                              caption="Patient-level gender analysis and consistency")
        
        # Age distribution
        if 'age' in df.columns:
            self.story.append(Paragraph("Age Distribution", self.heading2_style))
            
            age_stats = df['age'].describe()
            age_table = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'],
                'Value': [
                    age_stats['mean'],
                    age_stats['50%'],
                    age_stats['std'],
                    age_stats['min'],
                    age_stats['max'],
                    age_stats['25%'],
                    age_stats['75%']
                ]
            })
            age_table['Value'] = age_table['Value'].round(1)
            
            self.add_table(age_table, col_widths=[2.5*inch, 2.5*inch])
            
            age_viz_path = VIZ_DIR / 'eda' / 'age_distribution.png'
            if age_viz_path.exists():
                self.add_image(age_viz_path, width=6*inch, caption="Age distribution of patients")
        
        # Geographic distribution
        if 'mep_region' in df.columns:
            self.story.append(Paragraph("Geographic Distribution", self.heading2_style))
            
            mep_counts = df['mep_region'].value_counts().head(10)
            mep_table = pd.DataFrame({
                'MEP Region': mep_counts.index,
                'Record Count': mep_counts.values,
                'Percentage': (mep_counts.values / len(df) * 100).round(2)
            })
            
            self.add_table(mep_table, col_widths=[3*inch, 1.5*inch, 1.5*inch])
            
            mep_viz_path = VIZ_DIR / 'eda' / 'mep_regions.png'
            if mep_viz_path.exists():
                self.add_image(mep_viz_path, width=6*inch, caption="Geographic distribution by MEP region")
        
        self.story.append(PageBreak())
    
    def add_encounters_analysis(self):
        """Add patient encounters analysis section"""
        self.story.append(Paragraph("Patient Encounters Analysis", self.heading1_style))
        
        analysis_text = """
        Understanding healthcare utilization patterns is crucial for patient segmentation. This section 
        analyzes the frequency and distribution of patient encounters across different age groups, 
        providing insights into healthcare-seeking behavior and patient journey patterns.
        """
        self.story.append(Paragraph(analysis_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Encounters per patient
        encounters_per_patient = df.groupby('pid').size()
        
        encounter_stats = pd.DataFrame({
            'Statistic': ['Mean', 'Median', 'Std Dev', 'Min', 'Max', 'Q25', 'Q75'],
            'Encounters': [
                encounters_per_patient.mean(),
                encounters_per_patient.median(),
                encounters_per_patient.std(),
                encounters_per_patient.min(),
                encounters_per_patient.max(),
                encounters_per_patient.quantile(0.25),
                encounters_per_patient.quantile(0.75)
            ]
        })
        encounter_stats['Encounters'] = encounter_stats['Encounters'].round(2)
        
        self.story.append(Paragraph("Encounters per Patient Statistics", self.heading2_style))
        self.add_table(encounter_stats, col_widths=[2.5*inch, 2.5*inch])
        
        # Visualizations
        encounters_viz_path = VIZ_DIR / 'eda' / 'encounters_by_age_group.png'
        if encounters_viz_path.exists():
            self.story.append(Paragraph("Encounters by Age Group", self.heading2_style))
            self.add_image(encounters_viz_path, width=6*inch,
                          caption="Distribution of patient encounters across age groups")
        
        frequency_viz_path = VIZ_DIR / 'eda' / 'encounter_frequency_patterns.png'
        if frequency_viz_path.exists():
            self.story.append(Paragraph("Encounter Frequency Patterns", self.heading2_style))
            self.add_image(frequency_viz_path, width=6*inch,
                          caption="Frequency patterns of healthcare visits by age group")
        
        visits_time_path = VIZ_DIR / 'eda' / 'visits_over_time.png'
        if visits_time_path.exists():
            self.story.append(Paragraph("Visits Over Time", self.heading2_style))
            self.add_image(visits_time_path, width=6*inch,
                          caption="Temporal distribution of patient visits")
        
        self.story.append(PageBreak())
    
    def add_bp_analysis(self):
        """Add blood pressure analysis section"""
        self.story.append(Paragraph("Blood Pressure Analysis", self.heading1_style))
        
        analysis_text = """
        Blood pressure measurements are critical clinical variables for cardiovascular disease management 
        and patient segmentation. This section analyzes systolic and diastolic blood pressure patterns, 
        their relationships with BMI, and availability across different diagnostic codes.
        """
        self.story.append(Paragraph(analysis_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        # BP statistics
        if 'bp_systolic' in df.columns and 'bp_diastolic' in df.columns:
            bp_systolic_clean = pd.to_numeric(df['bp_systolic'], errors='coerce')
            bp_diastolic_clean = pd.to_numeric(df['bp_diastolic'], errors='coerce')
            
            bp_stats = pd.DataFrame({
                'Statistic': ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max'],
                'Systolic BP': [
                    bp_systolic_clean.notna().sum(),
                    bp_systolic_clean.mean(),
                    bp_systolic_clean.median(),
                    bp_systolic_clean.std(),
                    bp_systolic_clean.min(),
                    bp_systolic_clean.max()
                ],
                'Diastolic BP': [
                    bp_diastolic_clean.notna().sum(),
                    bp_diastolic_clean.mean(),
                    bp_diastolic_clean.median(),
                    bp_diastolic_clean.std(),
                    bp_diastolic_clean.min(),
                    bp_diastolic_clean.max()
                ]
            })
            
            bp_stats['Systolic BP'] = bp_stats['Systolic BP'].round(1)
            bp_stats['Diastolic BP'] = bp_stats['Diastolic BP'].round(1)
            
            self.story.append(Paragraph("Blood Pressure Statistics", self.heading2_style))
            self.add_table(bp_stats, col_widths=[2*inch, 2*inch, 2*inch])
        
        # BP vs BMI visualizations
        bp_bmi_path = VIZ_DIR / 'eda' / 'bp_vs_bmi_clean.png'
        if bp_bmi_path.exists():
            self.story.append(Paragraph("Systolic BP vs BMI", self.heading2_style))
            self.add_image(bp_bmi_path, width=6*inch,
                          caption="Relationship between systolic blood pressure and BMI")
        
        bp_diastolic_bmi_path = VIZ_DIR / 'eda' / 'bp_diastolic_vs_bmi_clean.png'
        if bp_diastolic_bmi_path.exists():
            self.story.append(Paragraph("Diastolic BP vs BMI", self.heading2_style))
            self.add_image(bp_diastolic_bmi_path, width=6*inch,
                          caption="Relationship between diastolic blood pressure and BMI")
        
        # BP by age group
        sbp_age_path = VIZ_DIR / 'eda' / 'sbp_by_age_group.png'
        if sbp_age_path.exists():
            self.story.append(Paragraph("Systolic BP by Age Group", self.heading2_style))
            self.add_image(sbp_age_path, width=6*inch,
                          caption="Systolic blood pressure distribution across age groups")
        
        # Top ICD3 codes with BP data
        if Path(VIZ_DIR / 'eda' / 'top10_icd3_bp_availability.csv').exists():
            top_icd_bp = pd.read_csv(VIZ_DIR / 'eda' / 'top10_icd3_bp_availability.csv')
            self.story.append(Paragraph("Top 10 ICD3 Codes with Most BP Data", self.heading2_style))
            
            # Rename columns for better readability and fit
            top_icd_bp_display = top_icd_bp.copy()
            top_icd_bp_display.columns = ['ICD3', 'N Rows', 'SBP Count', 'DBP Count', 
                                          'SBP %', 'DBP %', 'BP Total', 'SBP Med', 'DBP Med']
            
            # Round percentage columns
            if 'SBP %' in top_icd_bp_display.columns:
                top_icd_bp_display['SBP %'] = top_icd_bp_display['SBP %'].round(2)
            if 'DBP %' in top_icd_bp_display.columns:
                top_icd_bp_display['DBP %'] = top_icd_bp_display['DBP %'].round(2)
            
            # Column widths optimized for 9 columns to fit page width (6.5 inches total)
            # Page width is 8.5 - 1.5 (margins) = 7 inches, use 6.5 for safety
            col_widths = [0.6*inch, 0.65*inch, 0.7*inch, 0.7*inch, 0.65*inch, 
                         0.65*inch, 0.7*inch, 0.65*inch, 0.65*inch]
            
            # Use smaller font sizes for this wide table
            self.add_table(top_icd_bp_display, col_widths=col_widths, 
                          header_font_size=8, body_font_size=7)
        
        self.story.append(PageBreak())
    
    def add_icd_analysis(self, icd_code, icd_name):
        """Add ICD code-specific analysis section"""
        self.story.append(Paragraph(f"{icd_code} ({icd_name}) Analysis", self.heading1_style))
        
        # Load analysis CSV
        analysis_path = DATA_DIR / f'{icd_code.lower()}_clinical_variables_analysis.csv'
        if not analysis_path.exists():
            return
        
        analysis_df = pd.read_csv(analysis_path)
        
        # Summary statistics
        total_records = analysis_df.iloc[0][f'{icd_code}_Total']
        self.story.append(Paragraph(f"Total {icd_code} records: {int(total_records):,}", self.heading2_style))
        
        # Top 10 variables
        top10 = analysis_df.head(10)
        top10_display = top10[['Variable', f'{icd_code}_Percentage', 'All_Percentage', 'Difference']].copy()
        top10_display.columns = ['Variable', f'{icd_code} %', 'All %', 'Difference']
        top10_display = top10_display.round(2)
        
        self.story.append(Paragraph(f"Top 10 Clinical Variables - {icd_code} vs All Patients", self.heading2_style))
        self.add_table(top10_display, col_widths=[3*inch, 1*inch, 1*inch, 1*inch])
        
        # Visualizations
        comparison_path = VIZ_DIR / 'eda' / f'{icd_code.lower()}_clinical_variables_comparison.png'
        if comparison_path.exists():
            self.add_image(comparison_path, width=6*inch,
                          caption=f"Clinical variables availability: {icd_code} patients vs all patients")
        
        # ICD-specific visualizations
        if icd_code == 'I10':
            bp_dist_path = VIZ_DIR / 'eda' / 'i10_bp_distribution.png'
            if bp_dist_path.exists():
                self.story.append(Paragraph("Blood Pressure Distribution", self.heading2_style))
                self.add_image(bp_dist_path, width=6*inch,
                              caption="Blood pressure distribution for I10 (Hypertension) patients")
        
        elif icd_code == 'E11':
            bmi_bp_path = VIZ_DIR / 'eda' / 'e11_bmi_bp_distribution.png'
            if bmi_bp_path.exists():
                self.story.append(Paragraph("BMI and Blood Pressure Distribution", self.heading2_style))
                self.add_image(bmi_bp_path, width=6*inch,
                              caption="BMI and BP distributions for E11 (Type 2 Diabetes) patients")
        
        self.story.append(PageBreak())
    
    def add_findings_summary(self):
        """Add key findings summary section"""
        self.story.append(Paragraph("Key Findings Summary", self.heading1_style))
        
        findings = [
            ("Data Completeness", 
             "The dataset shows variable completeness across clinical measurements. Core identifiers (pid, specialty_name) are 100% complete, while clinical measurements range from 7-50% completeness depending on variable type and ICD code."),
            
            ("Anthropometric Data", 
             "Height, weight, and BMI measurements are available from both CV screening and physical examination sources. Missing data patterns correlate with patient age, with older patients showing different completeness patterns than younger cohorts."),
            
            ("Blood Pressure Data", 
             "BP measurements are available for approximately 31% of all records. Availability varies significantly by ICD code: I10 (Hypertension) patients show higher BP data availability, while E07 (Thyroid) patients have lower availability."),
            
            ("Patient Demographics", 
             f"Dataset contains {df['pid'].nunique():,} unique patients with consistent gender assignment across encounters. Age distribution spans the full lifespan with concentration in adult and elderly populations."),
            
            ("Healthcare Utilization", 
             "Patients show varying encounter frequencies, with most patients having single visits but a significant minority having multiple encounters over time. Encounter patterns vary by age group."),
            
            ("ICD Code-Specific Patterns", 
             "E07 (Thyroid) patients show high ultrasound data availability (49%) but lower BP data (7.7%). I10 (Hypertension) patients have high BP data availability. E11 (Diabetes) patients show diabetes-specific measurement availability and higher BMI/BP data relevance."),
            
            ("Data Quality Considerations", 
             "Missing data is not random - patterns vary by clinical variable type, patient age, and diagnostic code. Imputation strategies should account for these patterns. Multiple measurement sources (CV screening vs physical) provide redundancy for key variables.")
        ]
        
        for title, text in findings:
            self.story.append(Paragraph(f"<b>{title}</b>", self.heading2_style))
            self.story.append(Paragraph(text, self.body_style))
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(PageBreak())
    
    def add_recommendations(self):
        """Add recommendations for segmentation section"""
        self.story.append(Paragraph("Recommendations for Segmentation", self.heading1_style))
        
        self.story.append(Paragraph("Feature Selection", self.heading2_style))
        
        feature_text = """
        <b>Recommended Features for Segmentation:</b><br/><br/>
        
        1. <b>Demographics:</b> Age, gender, geographic region (MEP)<br/>
        2. <b>Clinical Measurements:</b> BMI, blood pressure (systolic/diastolic), pulse<br/>
        3. <b>Anthropometrics:</b> Height, weight, waist circumference (use canonical values)<br/>
        4. <b>Diagnostic Codes:</b> ICD3 codes (primary diagnosis)<br/>
        5. <b>Healthcare Utilization:</b> Number of encounters, time span between visits<br/>
        6. <b>Specialty:</b> Medical specialty associated with encounters<br/>
        7. <b>Clinical Variables by ICD:</b> Disease-specific measurements (e.g., diabetes data, ultrasound for thyroid)<br/><br/>
        
        <b>Canonicalization Strategy:</b> For variables with multiple sources (e.g., BMI from cv_screening7_bmi and physical3_bmi), 
        create canonical values using priority: prefer CV screening if available, otherwise use physical measurements.
        """
        self.story.append(Paragraph(feature_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        self.story.append(Paragraph("Missing Data Handling", self.heading2_style))
        
        missing_text = """
        <b>Strategies for Handling Missing Data:</b><br/><br/>
        
        1. <b>Missing Indicators:</b> Create binary flags for missing clinical variables to capture information about data availability<br/>
        2. <b>Conditional Imputation:</b> Use ICD code-specific imputation strategies (e.g., different imputation for E11 vs I10)<br/>
        3. <b>Age-Stratified Imputation:</b> Account for age-dependent missing patterns in anthropometric data<br/>
        4. <b>Domain-Specific Imputation:</b> Use clinical knowledge (e.g., normal BP ranges by age) for imputation<br/>
        5. <b>Multiple Imputation:</b> Consider multiple imputation for key variables to account for uncertainty<br/><br/>
        
        <b>Thresholds:</b> Consider excluding variables with <5% completeness or creating separate models for high/low completeness segments.
        """
        self.story.append(Paragraph(missing_text, self.body_style))
        self.story.append(Spacer(1, 0.2*inch))
        
        self.story.append(Paragraph("Segmentation Approach Recommendations", self.heading2_style))
        
        segmentation_text = """
        <b>Recommended Segmentation Strategies:</b><br/><br/>
        
        1. <b>ICD Code-Based Segmentation:</b> Start with primary ICD3 code segmentation (E07, I10, E11, etc.) as these represent distinct clinical conditions<br/>
        2. <b>Hybrid Approach:</b> Combine ICD codes with clinical variables (BMI, BP, age) for sub-segmentation<br/>
        3. <b>Multi-Level Segmentation:</b> 
           - Level 1: ICD code groups
           - Level 2: Clinical severity (e.g., BP levels, BMI categories)
           - Level 3: Utilization patterns (high vs low utilizers)<br/>
        4. <b>Patient-Level Aggregation:</b> Aggregate encounter-level data to patient-level features:
           - Use baseline/most recent values for clinical measurements
           - Count encounters, calculate time spans
           - Identify primary ICD codes per patient<br/>
        5. <b>Unsupervised Clustering:</b> After ICD-based segmentation, apply clustering (K-means, hierarchical) within each ICD group using clinical variables<br/>
        6. <b>Validation:</b> Validate segments using:
           - Clinical interpretability
           - Statistical distinctiveness
           - Healthcare utilization patterns
           - Outcomes (if available)<br/><br/>
        
        <b>Next Steps:</b>
        • Build patient-level feature table with canonicalized variables
        • Create ICD code-specific cohorts
        • Perform clustering within each ICD code group
        • Validate and interpret segments
        • Document segment characteristics and clinical implications
        """
        self.story.append(Paragraph(segmentation_text, self.body_style))
        
        self.story.append(PageBreak())
    
    def generate(self):
        """Generate the complete PDF report"""
        print("Generating PDF report...")
        
        # Add all sections
        self.add_title_page()
        self.add_executive_summary()
        self.add_dataset_overview()
        self.add_data_quality()
        self.add_anthropometric_analysis()
        self.add_demographics()
        self.add_encounters_analysis()
        self.add_bp_analysis()
        self.add_icd_analysis('E07', 'Thyroid Disorders')
        self.add_icd_analysis('I10', 'Essential/Primary Hypertension')
        self.add_icd_analysis('E11', 'Type 2 Diabetes Mellitus')
        self.add_findings_summary()
        self.add_recommendations()
        
        # Build PDF
        self.doc.build(self.story)
        print(f"PDF report generated successfully: {self.output_path}")


if __name__ == "__main__":
    output_path = REPORTS_DIR / 'eda_report.pdf'
    generator = PDFReportGenerator(output_path)
    generator.generate()

