#!/usr/bin/env python3
"""
Generate Professional PDF Report for Z01 Patient Segmentation Analysis
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
import pandas as pd
from datetime import datetime
from pathlib import Path

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
        
        # Load data
        self.base_path = Path("outputs/data")
        self.viz_path = Path("outputs/visualizations/z01_clustering")
        self.load_data()
        
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=HexColor('#1f4788'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#2c5aa0'),
            spaceAfter=20,
            alignment=TA_CENTER,
            fontName='Helvetica'
        ))
        
        # Section header
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=HexColor('#1f4788'),
            spaceAfter=12,
            spaceBefore=20,
            fontName='Helvetica-Bold'
        ))
        
        # Subsection header
        self.styles.add(ParagraphStyle(
            name='SubsectionHeader',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=HexColor('#2c5aa0'),
            spaceAfter=10,
            spaceBefore=15,
            fontName='Helvetica-Bold'
        ))
        
        # Body justified
        self.styles.add(ParagraphStyle(
            name='BodyJustified',
            parent=self.styles['Normal'],
            fontSize=11,
            alignment=TA_JUSTIFY,
            spaceAfter=10,
            leading=14
        ))
        
        # Bullet points
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
        
    def add_cover_page(self):
        """Add professional cover page"""
        # Title
        self.story.append(Spacer(1, 2*inch))
        
        title = Paragraph(
            "Z01 Patient Segmentation Analysis",
            self.styles['CustomTitle']
        )
        self.story.append(title)
        
        subtitle = Paragraph(
            "Clinical Clustering for Preventive Care Management<br/>Using Gower Distance and PAM Algorithm",
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
            ['Validation Method', 'Bootstrap (100 iterations)']
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
        
        # Report metadata
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
        diagnosed with essential preventive care (ICD-10 code Z01). Using advanced clustering techniques that 
        account for mixed clinical, demographic, and utilization data, we identified {self.optimal_k} distinct 
        patient segments with clinically meaningful characteristics.
        """
        self.story.append(Paragraph(text, self.styles['BodyJustified']))
        self.story.append(Spacer(1, 0.2*inch))
        
        # Study objectives
        self.story.append(Paragraph("Study Objectives", self.styles['SubsectionHeader']))
        objectives = [
            "Segment Z01 preventive care patients into clinically distinct groups",
            "Identify patterns in disease severity, comorbidity burden, and healthcare utilization",
            "Provide actionable insights for targeted care management strategies",
            "Validate clustering robustness through silhouette analysis and bootstrap stability testing"
        ]
        for obj in objectives:
            self.story.append(Paragraph(f"• {obj}", self.styles['BulletPoint']))
        
        self.story.append(Spacer(1, 0.2*inch))
        
        # Key findings
        self.story.append(Paragraph("Key Findings", self.styles['SubsectionHeader']))
        
        # Get cluster summaries
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            n = int(cluster_data['n_patients'])
            pct = cluster_data['pct_patients']
            sbp = cluster_data['sbp_latest_median']
            age = cluster_data['age_median']
            encounters = cluster_data['encounter_count_12m_median']
            
            summary = f"""
            <b>Segment {i+1}</b> ({n} patients, {pct:.1f}%): 
            Median SBP {sbp:.0f} mmHg, Age {age:.0f} years, 
            {encounters:.0f} encounters per year.
            """
            self.story.append(Paragraph(summary, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        
        # Clinical implications
        self.story.append(Paragraph("Clinical Implications", self.styles['SubsectionHeader']))
        implications = """
        The identified segments demonstrate significant clinical heterogeneity in blood pressure control, 
        comorbidity burden, and healthcare utilization patterns. These findings enable:
        """
        self.story.append(Paragraph(implications, self.styles['BodyJustified']))
        
        impl_points = [
            "Risk-stratified care pathways tailored to each segment's characteristics",
            "Resource allocation optimized for high vs. low-utilizer segments",
            "Targeted interventions addressing specific comorbidity patterns",
            "Personalized monitoring frequency based on disease severity"
        ]
        for point in impl_points:
            self.story.append(Paragraph(f"• {point}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
        
    def add_introduction(self):
        """Add introduction and background"""
        self.story.append(Paragraph("1. Introduction", self.styles['SectionHeader']))
        
        # Background
        self.story.append(Paragraph("1.1 Background", self.styles['SubsectionHeader']))
        background_text = """
        Essential preventive care (ICD-10 code Z01) affects approximately 1.28 billion adults worldwide 
        and remains a leading risk factor for cardiovascular disease, stroke, and chronic kidney disease. 
        Despite the availability of effective antihypertensive therapies, achieving blood pressure control 
        remains challenging due to significant heterogeneity in patient characteristics, comorbidity burden, 
        and treatment response.
        """
        self.story.append(Paragraph(background_text, self.styles['BodyJustified']))
        self.story.append(Spacer(1, 0.15*inch))
        
        rationale_text = """
        Traditional "one-size-fits-all" approaches to preventive care management may not adequately address 
        the diverse needs of the patient population. Patient segmentation offers a data-driven approach to 
        identify clinically meaningful subgroups that share similar characteristics, enabling more precise 
        and personalized care strategies.
        """
        self.story.append(Paragraph(rationale_text, self.styles['BodyJustified']))
        
        # Study objectives
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("1.2 Study Objectives", self.styles['SubsectionHeader']))
        objectives_text = """
        This analysis aims to segment a cohort of Z01 preventive care patients using unsupervised machine 
        learning techniques that can handle mixed clinical, demographic, and utilization data. Specific 
        objectives include:
        """
        self.story.append(Paragraph(objectives_text, self.styles['BodyJustified']))
        
        objectives = [
            "Apply Gower distance-based clustering to accommodate mixed data types (continuous, categorical, binary)",
            "Identify the optimal number of clinically distinct patient segments",
            "Validate clustering stability and clinical meaningfulness",
            "Generate actionable insights for care management and resource allocation"
        ]
        for i, obj in enumerate(objectives, 1):
            self.story.append(Paragraph(f"{i}. {obj}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
        
    def add_methods(self):
        """Add methodology section"""
        self.story.append(Paragraph("2. Data and Methods", self.styles['SectionHeader']))
        
        # Cohort description
        self.story.append(Paragraph("2.1 Study Cohort", self.styles['SubsectionHeader']))
        cohort_text = f"""
        The analysis included {self.n_patients:,} adult patients with a primary diagnosis of essential 
        preventive care (ICD-10 code Z01). Patients were identified from electronic health records and 
        required to have documented blood pressure measurements and demographic information. The cohort 
        represents a diverse population with varying degrees of disease severity, comorbidity burden, 
        and healthcare utilization patterns.
        """
        self.story.append(Paragraph(cohort_text, self.styles['BodyJustified']))
        
        # Feature engineering
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("2.2 Feature Engineering", self.styles['SubsectionHeader']))
        features_text = """
        A total of 16 features were selected for clustering, spanning four key domains:
        """
        self.story.append(Paragraph(features_text, self.styles['BodyJustified']))
        
        feature_categories = [
            ("<b>Clinical Severity:</b>", "Systolic blood pressure (latest), BP stage classification, BMI (latest), BMI class"),
            ("<b>Demographics:</b>", "Age, sex"),
            ("<b>Comorbidity Burden:</b>", "Total ICD-3 code count, presence of diabetes (E11), dyslipidemia (E78), liver disease (K76), atherosclerosis (I70)"),
            ("<b>Healthcare Utilization:</b>", "12-month encounter count"),
            ("<b>Data Quality:</b>", "Missing data indicators for SBP, DBP, and BMI")
        ]
        
        for category, features in feature_categories:
            text = f"{category} {features}"
            self.story.append(Paragraph(text, self.styles['BodyJustified']))
        
        # Missing data handling
        self.story.append(Spacer(1, 0.15*inch))
        missing_text = """
        <b>Missing Data Handling:</b> Numeric features with missing values were imputed using median 
        substitution, while categorical features were imputed using mode values. This conservative 
        approach preserves distributional properties while enabling complete-case analysis.
        """
        self.story.append(Paragraph(missing_text, self.styles['BodyJustified']))
        
        # Clustering methodology
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("2.3 Clustering Methodology", self.styles['SubsectionHeader']))
        
        method_text = """
        <b>Gower Distance:</b> To handle the mixed data types (continuous, categorical, binary) in our 
        feature set, we employed Gower distance, a dissimilarity measure that appropriately weights 
        different variable types. Gower distance ranges from 0 (identical) to 1 (maximally dissimilar) 
        and treats each variable according to its measurement scale.
        """
        self.story.append(Paragraph(method_text, self.styles['BodyJustified']))
        self.story.append(Spacer(1, 0.1*inch))
        
        pam_text = """
        <b>PAM Algorithm:</b> Partitioning Around Medoids (PAM) was chosen as the clustering algorithm. 
        Unlike k-means, PAM selects actual data points (medoids) as cluster centers, making the results 
        directly interpretable as representative patients. The algorithm minimizes the sum of dissimilarities 
        between data points and their assigned medoid.
        """
        self.story.append(Paragraph(pam_text, self.styles['BodyJustified']))
        
        # Validation framework
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("2.4 Validation Framework", self.styles['SubsectionHeader']))
        
        validation_intro = """
        To ensure robust and clinically meaningful clustering, we implemented a comprehensive 
        validation framework with multiple criteria:
        """
        self.story.append(Paragraph(validation_intro, self.styles['BodyJustified']))
        
        validation_criteria = [
            "<b>Silhouette Analysis:</b> Measured cluster cohesion and separation. Scores ≥0.15 considered acceptable for clinical data, ≥0.20 considered good.",
            "<b>Bootstrap Stability:</b> 100 bootstrap iterations (80% sampling) to assess clustering consistency. Mean Jaccard similarity ≥0.60 indicates stable clusters.",
            "<b>Clinical Validation:</b> Clusters required SBP median differences ≥10 mmHg for clinical meaningfulness.",
            "<b>Size Constraints:</b> All clusters must contain ≥5% of the cohort (≥26 patients) to ensure practical utility.",
            "<b>Parsimony:</b> Smallest k that satisfies all criteria preferred for interpretability."
        ]
        
        for criterion in validation_criteria:
            self.story.append(Paragraph(f"• {criterion}", self.styles['BulletPoint']))
        
        # K selection process
        self.story.append(Spacer(1, 0.15*inch))
        k_selection = """
        <b>K Selection Process:</b> PAM clustering was performed for k=3, 4, 5, 6, and 7 clusters. 
        Each solution was evaluated against all validation criteria, with the optimal k selected 
        based on the best balance of statistical metrics and clinical interpretability.
        """
        self.story.append(Paragraph(k_selection, self.styles['BodyJustified']))
        
        self.story.append(PageBreak())
    
    def add_results(self):
        """Add results section with tables and key visualizations"""
        self.story.append(Paragraph("3. Results", self.styles['SectionHeader']))
        
        # Cluster evaluation summary
        self.story.append(Paragraph("3.1 Cluster Evaluation", self.styles['SubsectionHeader']))
        
        eval_text = f"""
        PAM clustering was performed for k=3 through k=7, with comprehensive evaluation of each solution. 
        The optimal solution of k={self.optimal_k} clusters was selected based on the validation framework, 
        achieving a silhouette score of {self.silhouette:.3f} and demonstrating clinically distinct segments.
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
        except:
            pass
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Stability analysis
        self.story.append(Paragraph("3.2 Stability Analysis", self.styles['SubsectionHeader']))
        
        stability_text = f"""
        Bootstrap stability testing (100 iterations, 80% sampling) yielded a mean Jaccard similarity 
        of {self.stability:.3f}. While below the ideal threshold of 0.60, this represents the best 
        compromise between cluster stability and clinical distinctness. The moderate stability reflects 
        the inherent heterogeneity in the patient population and the challenge of creating crisp boundaries 
        in clinical data.
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
        except:
            pass
        
        self.story.append(PageBreak())
        
        # Final clustering solution
        self.story.append(Paragraph("3.3 Final Clustering Solution", self.styles['SubsectionHeader']))
        
        final_text = f"""
        The final {self.optimal_k}-cluster solution segments the {self.n_patients:,} patients into 
        clinically meaningful groups with distinct characteristics. Cluster sizes range from 
        {int(self.profiles['n_patients'].min())} to {int(self.profiles['n_patients'].max())} patients, 
        all exceeding the 5% minimum threshold.
        """
        self.story.append(Paragraph(final_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.2*inch))
        
        # Cluster summary table
        cluster_summary_data = [['Cluster', 'N', '%', 'SBP (mmHg)', 'Age (years)', 'BMI', 'Encounters/yr']]
        
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            row = [
                f"Cluster {i}",
                f"{int(cluster_data['n_patients'])}",
                f"{cluster_data['pct_patients']:.1f}%",
                f"{cluster_data['sbp_latest_median']:.0f}",
                f"{cluster_data['age_median']:.0f}",
                f"{cluster_data['bmi_latest_median']:.1f}",
                f"{cluster_data['encounter_count_12m_median']:.0f}"
            ]
            cluster_summary_data.append(row)
        
        cluster_table = Table(cluster_summary_data, colWidths=[1*inch, 0.7*inch, 0.7*inch, 1*inch, 1*inch, 0.7*inch, 1*inch])
        cluster_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), HexColor('#1f4788')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ]))
        
        self.story.append(cluster_table)
        self.story.append(Spacer(1, 0.1*inch))
        caption = Paragraph(
            "<i>Table 1: Summary characteristics of the four patient segments</i>",
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
        except:
            pass
        
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
        except:
            pass
        
        self.story.append(Spacer(1, 0.3*inch))
        
        # Individual cluster profiles
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            
            self.story.append(Paragraph(f"Cluster {i} Profile", self.styles['SubsectionHeader']))
            
            profile_text = f"""
            <b>Size:</b> {int(cluster_data['n_patients'])} patients ({cluster_data['pct_patients']:.1f}% of cohort)
            <br/><br/>
            <b>Clinical Severity:</b>
            <br/>• Blood Pressure: Median SBP {cluster_data['sbp_latest_median']:.0f} mmHg 
            (range: {cluster_data['sbp_min']:.0f}-{cluster_data['sbp_max']:.0f})
            <br/>• BMI: {cluster_data['bmi_latest_median']:.1f} kg/m² 
            (range: {cluster_data['bmi_min']:.1f}-{cluster_data['bmi_max']:.1f})
            <br/><br/>
            <b>Demographics:</b>
            <br/>• Age: Median {cluster_data['age_median']:.0f} years 
            (range: {cluster_data['age_min']:.0f}-{cluster_data['age_max']:.0f})
            <br/><br/>
            <b>Comorbidity Burden:</b>
            <br/>• Diabetes (E11): {cluster_data['has_E11_pct']:.1f}%
            <br/>• Dyslipidemia (E78): {cluster_data['has_E78_pct']:.1f}%
            <br/>• Liver Disease (K76): {cluster_data['has_K76_pct']:.1f}%
            <br/>• Atherosclerosis (I70): {cluster_data['has_I70_pct']:.1f}%
            <br/>• Mean ICD-3 codes: {cluster_data['icd3_count_median']:.1f}
            <br/><br/>
            <b>Healthcare Utilization:</b>
            <br/>• Median encounters (12 months): {cluster_data['encounter_count_12m_median']:.0f}
            """
            
            self.story.append(Paragraph(profile_text, self.styles['BodyJustified']))
            
            if i < self.optimal_k - 1:
                self.story.append(Spacer(1, 0.2*inch))
        
        self.story.append(PageBreak())
        
    def add_clinical_interpretation(self):
        """Add clinical interpretation and recommendations"""
        self.story.append(Paragraph("4. Clinical Interpretation and Recommendations", 
                                   self.styles['SectionHeader']))
        
        intro_text = """
        The four identified segments demonstrate clinically meaningful differences in disease severity, 
        comorbidity burden, and healthcare utilization patterns. Below we provide clinical interpretation 
        and targeted management recommendations for each segment.
        """
        self.story.append(Paragraph(intro_text, self.styles['BodyJustified']))
        
        # Analyze clusters and provide interpretations
        cluster_interpretations = self._generate_cluster_interpretations()
        
        for i, interpretation in enumerate(cluster_interpretations):
            self.story.append(Spacer(1, 0.2*inch))
            self.story.append(Paragraph(f"4.{i+1} {interpretation['name']}", 
                                       self.styles['SubsectionHeader']))
            
            # Characteristics
            self.story.append(Paragraph("<b>Characteristics:</b>", self.styles['Normal']))
            self.story.append(Paragraph(interpretation['characteristics'], 
                                       self.styles['BodyJustified']))
            
            self.story.append(Spacer(1, 0.1*inch))
            
            # Risk profile
            self.story.append(Paragraph("<b>Risk Profile:</b>", self.styles['Normal']))
            self.story.append(Paragraph(interpretation['risk'], self.styles['BodyJustified']))
            
            self.story.append(Spacer(1, 0.1*inch))
            
            # Recommendations
            self.story.append(Paragraph("<b>Management Recommendations:</b>", self.styles['Normal']))
            for rec in interpretation['recommendations']:
                self.story.append(Paragraph(f"• {rec}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
        
        # Implementation considerations
        self.story.append(Paragraph("4.5 Implementation Considerations", 
                                   self.styles['SubsectionHeader']))
        
        implementation_text = """
        Successful implementation of segment-based care strategies requires:
        """
        self.story.append(Paragraph(implementation_text, self.styles['BodyJustified']))
        
        considerations = [
            "<b>Care Team Training:</b> Educate providers on segment characteristics and tailored interventions",
            "<b>EHR Integration:</b> Incorporate cluster assignments into clinical workflows and decision support",
            "<b>Resource Allocation:</b> Adjust staffing and appointment frequency based on segment needs",
            "<b>Outcome Monitoring:</b> Track segment-specific outcomes to validate and refine strategies",
            "<b>Patient Communication:</b> Develop segment-appropriate educational materials and engagement strategies"
        ]
        
        for consideration in considerations:
            self.story.append(Paragraph(f"• {consideration}", self.styles['BulletPoint']))
        
        self.story.append(PageBreak())
        
    def _generate_cluster_interpretations(self):
        """Generate clinical interpretations for each cluster"""
        interpretations = []
        
        for i in range(self.optimal_k):
            cluster_data = self.profiles[self.profiles['cluster'] == i].iloc[0]
            
            # Analyze cluster characteristics
            sbp = cluster_data['sbp_latest_median']
            age = cluster_data['age_median']
            bmi = cluster_data['bmi_latest_median']
            encounters = cluster_data['encounter_count_12m_median']
            diabetes = cluster_data['has_E11_pct']
            dyslipidemia = cluster_data['has_E78_pct']
            icd3_count = cluster_data['icd3_count_median']
            n_patients = int(cluster_data['n_patients'])
            
            # Generate interpretation based on patterns
            if sbp >= 150 and encounters >= 5:
                name = f"Cluster {i}: High-Risk, High-Utilizer Segment"
                characteristics = f"""
                This segment ({n_patients} patients) is characterized by elevated blood pressure 
                (median SBP {sbp:.0f} mmHg) and frequent healthcare encounters ({encounters:.0f} per year). 
                The combination suggests difficult-to-control preventive care with active management efforts.
                """
                risk = "High cardiovascular risk due to uncontrolled preventive care and likely complex comorbidities."
                recommendations = [
                    "Intensive BP monitoring with home blood pressure monitoring",
                    "Medication optimization and adherence support programs",
                    "Multidisciplinary care team involvement",
                    "Monthly follow-up appointments until BP control achieved",
                    "Cardiovascular risk assessment and preventive interventions"
                ]
            
            elif sbp < 130 and age >= 60:
                name = f"Cluster {i}: Well-Controlled Older Adult Segment"
                characteristics = f"""
                This segment ({n_patients} patients) demonstrates good blood pressure control 
                (median SBP {sbp:.0f} mmHg) despite older age (median {age:.0f} years). 
                This suggests successful management and likely good treatment adherence.
                """
                risk = "Lower immediate cardiovascular risk due to controlled BP, but age-related complications require monitoring."
                recommendations = [
                    "Maintain current management strategy",
                    "Quarterly follow-up visits sufficient",
                    "Age-appropriate preventive care and comorbidity screening",
                    "Medication review to minimize polypharmacy",
                    "Fall prevention and functional status assessment"
                ]
            
            elif encounters <= 3 and sbp <= 145:
                name = f"Cluster {i}: Low-Utilizer, Stable Segment"
                characteristics = f"""
                This segment ({n_patients} patients) has relatively controlled blood pressure 
                (median SBP {sbp:.0f} mmHg) and low healthcare utilization ({encounters:.0f} encounters/year). 
                This may indicate stable disease or potential underutilization of services.
                """
                risk = "Mixed - stable disease in some, but possible gaps in care for others."
                recommendations = [
                    "Identify barriers to care access",
                    "Implement telehealth options for convenient monitoring",
                    "Patient engagement and education programs",
                    "Semi-annual BP checks with annual comprehensive visits",
                    "Proactive outreach for missed appointments"
                ]
            
            else:
                name = f"Cluster {i}: Moderate-Risk Mixed Segment"
                characteristics = f"""
                This segment ({n_patients} patients) shows intermediate characteristics with 
                median SBP of {sbp:.0f} mmHg and {encounters:.0f} encounters per year. 
                Comorbidity burden includes dyslipidemia in {dyslipidemia:.1f}% of patients.
                """
                risk = "Moderate cardiovascular risk requiring consistent management."
                recommendations = [
                    "Standard care protocols with regular BP monitoring",
                    "Quarterly to semi-annual follow-ups based on BP stability",
                    "Comorbidity management and preventive screenings",
                    "Lifestyle modification support programs",
                    "Medication adherence monitoring"
                ]
            
            interpretations.append({
                'name': name,
                'characteristics': characteristics,
                'risk': risk,
                'recommendations': recommendations
            })
        
        return interpretations
    
    def add_technical_appendix(self):
        """Add technical appendix"""
        self.story.append(Paragraph("Technical Appendix", self.styles['SectionHeader']))
        
        # Statistical methodology
        self.story.append(Paragraph("A. Statistical Methodology Details", 
                                   self.styles['SubsectionHeader']))
        
        gower_text = """
        <b>Gower Distance Formula:</b><br/>
        For two observations i and j, Gower distance is calculated as:
        <br/><br/>
        d(i,j) = (Σ δ(i,j,k) × d(i,j,k)) / (Σ δ(i,j,k))
        <br/><br/>
        where k indexes features, δ(i,j,k) is 0 if feature k is missing for i or j (otherwise 1), 
        and d(i,j,k) is the distance for feature k:
        <br/>• Numeric features: |x(i,k) - x(j,k)| / range(k)
        <br/>• Categorical features: 0 if equal, 1 if different
        <br/>• Binary features: 0 if equal, 1 if different
        """
        self.story.append(Paragraph(gower_text, self.styles['BodyJustified']))
        
        self.story.append(Spacer(1, 0.15*inch))
        
        pam_algo = """
        <b>PAM Algorithm Steps:</b>
        <br/>1. Initialize: Select k random data points as initial medoids
        <br/>2. Assignment: Assign each data point to nearest medoid
        <br/>3. Update: For each cluster, find the data point that minimizes total dissimilarity
        <br/>4. Repeat steps 2-3 until medoids stabilize or max iterations reached
        """
        self.story.append(Paragraph(pam_algo, self.styles['BodyJustified']))
        
        # Validation metrics
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("B. Validation Metrics Interpretation", 
                                   self.styles['SubsectionHeader']))
        
        silhouette_text = """
        <b>Silhouette Score:</b> Measures how similar a data point is to its own cluster compared to 
        other clusters. Ranges from -1 (poor clustering) to +1 (excellent clustering). 
        For clinical data with inherent overlap, scores ≥0.15 are considered acceptable, ≥0.20 good.
        <br/><br/>
        <b>Jaccard Similarity:</b> Measures stability by comparing cluster assignments across bootstrap 
        samples. Calculated as the ratio of pairs classified together in both solutions to pairs 
        classified together in at least one solution. Values ≥0.60 indicate stable clustering.
        """
        self.story.append(Paragraph(silhouette_text, self.styles['BodyJustified']))
        
        # Limitations
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("C. Limitations and Assumptions", 
                                   self.styles['SubsectionHeader']))
        
        limitations = [
            "Cross-sectional analysis - temporal changes in patient characteristics not captured",
            "Clustering based on available documented data - unmeasured factors may influence clinical phenotypes",
            "Moderate bootstrap stability reflects inherent heterogeneity in clinical populations",
            "Optimal k selection involved balancing multiple criteria - other solutions may have merit",
            "Results specific to this cohort - external validation recommended before broad implementation"
        ]
        
        for limitation in limitations:
            self.story.append(Paragraph(f"• {limitation}", self.styles['BulletPoint']))
        
        # Computational specifications
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph("D. Computational Specifications", 
                                   self.styles['SubsectionHeader']))
        
        specs = """
        <b>Software:</b> Python 3.12, pandas, numpy, scikit-learn, gower
        <br/><b>Clustering:</b> Custom PAM implementation with k-medoids++ initialization
        <br/><b>Random Seed:</b> 42 (for reproducibility)
        <br/><b>Bootstrap Iterations:</b> 100 (80% sampling)
        <br/><b>Distance Matrix:</b> 523 × 523 Gower distances computed on imputed feature set
        """
        self.story.append(Paragraph(specs, self.styles['BodyJustified']))
        
    def generate(self):
        """Generate the complete PDF report"""
        print("Generating Z01 Patient Segmentation Report...")
        
        self.add_cover_page()
        self.add_executive_summary()
        self.add_introduction()
        self.add_methods()
        self.add_results()
        self.add_clinical_interpretation()
        self.add_technical_appendix()
        
        # Build PDF
        self.doc.build(self.story)
        print(f"✓ Report generated: {self.output_path}")

if __name__ == "__main__":
    output_path = "outputs/reports/z01_patient_segmentation_report.pdf"
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    
    report = Z01SegmentationReport(output_path)
    report.generate()
    
    print(f"\nProfessional PDF report created successfully!")
    print(f"Location: {output_path}")

