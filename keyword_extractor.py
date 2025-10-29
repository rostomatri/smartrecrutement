"""
Keyword Extraction Module
Extracts important keywords from text using NLP techniques
"""

import re
from typing import List, Set, Dict
from collections import Counter
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer


class KeywordExtractor:
    """Extract keywords from text using various NLP techniques"""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the keyword extractor
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP processing
        """
        self.use_spacy = use_spacy
        self.nlp = None
        
        if use_spacy:
            try:
                # Try to load the medium model first (better for entity recognition)
                self.nlp = spacy.load("en_core_web_md")
                print("[OK] Loaded spaCy model: en_core_web_md")
            except OSError:
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                    print("[OK] Loaded spaCy model: en_core_web_sm")
                except OSError:
                    print("[WARNING] spaCy model not found. Install with: python -m spacy download en_core_web_md")
                    self.use_spacy = False
    
    def preprocess_text(self, text: str) -> str:
        """
        Clean and preprocess text
        
        Args:
            text: Raw text to preprocess
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^a-zA-Z0-9\s\+\#]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def extract_keywords_spacy(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract keywords using spaCy NLP
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of extracted keywords
        """
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        
        # Extract nouns, proper nouns, and important entities
        keywords = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'SKILL', 'GPE']:
                keywords.append(ent.text.lower())
        
        # Extract nouns and adjectives
        for token in doc:
            if token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and not token.is_stop and len(token.text) > 2:
                keywords.append(token.lemma_.lower())
        
        # Count frequency and return top keywords
        keyword_freq = Counter(keywords)
        return [kw for kw, _ in keyword_freq.most_common(top_n)]
    
    def extract_keywords_tfidf(self, text: str, top_n: int = 20) -> List[str]:
        """
        Extract keywords using TF-IDF
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of extracted keywords
        """
        # Preprocess
        cleaned_text = self.preprocess_text(text)
        
        # Custom stop words - remove common but not meaningful words
        custom_stop_words = [
            'company', 'city', 'state', 'location', 'email', 'phone', 'address',
            'january', 'february', 'march', 'april', 'may', 'june', 'july',
            'august', 'september', 'october', 'november', 'december',
            'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun',
            '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12'
        ]
        
        # Use TF-IDF
        vectorizer = TfidfVectorizer(
            max_features=top_n * 2,  # Get more features to filter
            stop_words='english',
            ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
            min_df=1,
            max_df=0.95
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform([cleaned_text])
            feature_names = vectorizer.get_feature_names_out()
            
            # Get TF-IDF scores
            scores = tfidf_matrix.toarray()[0]
            
            # Sort by score
            keyword_scores = list(zip(feature_names, scores))
            keyword_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Filter out custom stop words and return top keywords
            filtered_keywords = [
                kw for kw, score in keyword_scores 
                if kw not in custom_stop_words and len(kw) > 2
            ]
            
            return filtered_keywords[:top_n]
        except:
            return []
    
    def extract_technical_skills(self, text: str) -> Set[str]:
        """
        Extract common technical skills and programming languages
        
        Args:
            text: Text to extract skills from
            
        Returns:
            Set of detected skills
        """
        text_lower = text.lower()
        
        # Comprehensive technical skills and tools database
        skills_database = {
            # Programming languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'c ', 'ruby', 'go', 'golang', 'rust',
            'php', 'swift', 'kotlin', 'scala', 'r', 'matlab', 'sql', 'perl', 'bash', 'powershell',
            'vba', 'sas', 'julia', 'dart', 'objective-c',
            
            # Web technologies
            'html', 'html5', 'css', 'css3', 'react', 'reactjs', 'angular', 'angularjs', 'vue', 'vuejs',
            'node.js', 'nodejs', 'django', 'flask', 'spring', 'spring boot', 'express', 'expressjs',
            'fastapi', 'next.js', 'nextjs', 'gatsby', 'jquery', 'bootstrap', 'tailwind',
            'asp.net', 'laravel', 'ruby on rails', 'svelte',
            
            # Databases & Data Storage
            'mysql', 'postgresql', 'postgres', 'mongodb', 'redis', 'elasticsearch', 'oracle',
            'dynamodb', 'cassandra', 'neo4j', 'sqlite', 'mariadb', 'microsoft sql server',
            'sql server', 'couchdb', 'firebase', 'snowflake', 'bigquery', 'redshift',
            
            # Cloud & DevOps
            'aws', 'amazon web services', 'azure', 'microsoft azure', 'gcp', 'google cloud',
            'docker', 'kubernetes', 'k8s', 'jenkins', 'gitlab', 'github actions',
            'terraform', 'ansible', 'ci/cd', 'circleci', 'travis ci', 'cloudformation',
            'vagrant', 'puppet', 'chef', 'bamboo',
            
            # Data Science & ML & Analytics
            'machine learning', 'deep learning', 'nlp', 'natural language processing',
            'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'sklearn', 'pandas', 'numpy',
            'spark', 'apache spark', 'hadoop', 'pyspark', 'jupyter', 'tableau', 'power bi',
            'looker', 'data analysis', 'data analytics', 'data visualization', 'data mining',
            'statistical analysis', 'predictive modeling', 'forecasting', 'time series',
            'regression', 'classification', 'clustering', 'neural networks', 'computer vision',
            'image processing', 'opencv', 'data warehousing', 'etl', 'big data',
            'business intelligence', 'analytics', 'quantitative analysis',
            
            # Operations & Business
            'operations management', 'process optimization', 'supply chain', 'inventory management',
            'logistics', 'lean', 'six sigma', 'kaizen', 'project management', 'agile', 'scrum',
            'kanban', 'waterfall', 'business analysis', 'business process', 'kpi', 'metrics',
            'performance management', 'quality assurance', 'quality control', 'continuous improvement',
            
            # Version Control & Collaboration
            'git', 'github', 'gitlab', 'bitbucket', 'svn', 'mercurial', 'version control',
            
            # Testing
            'unit testing', 'integration testing', 'selenium', 'pytest', 'junit', 'jest',
            'testing', 'test automation', 'qa', 'tdd', 'bdd',
            
            # Other technical tools
            'linux', 'unix', 'windows server', 'jira', 'confluence', 'slack', 'teams',
            'rest api', 'restful', 'graphql', 'soap', 'microservices', 'api',
            'json', 'xml', 'yaml', 'grpc', 'websocket', 'oauth', 'jwt',
            'excel', 'microsoft excel', 'google sheets', 'vba', 'macros',
            'powerpoint', 'word', 'office 365', 'google workspace',
            
            # Soft Skills & Methods (important for operations/analytics roles)
            'leadership', 'cross-functional', 'stakeholder management', 'communication',
            'problem solving', 'critical thinking', 'decision making', 'strategic planning',
            'change management', 'vendor management', 'budget management',
            'root cause analysis', 'swot analysis', 'gap analysis',
            
            # Specific methodologies
            'agile methodology', 'scrum methodology', 'devops', 'devsecops',
            'continuous integration', 'continuous deployment', 'automation',
        }
        
        detected_skills = set()
        
        # First pass: exact phrase matching (for multi-word skills)
        for skill in skills_database:
            if ' ' in skill:  # Multi-word skills
                if skill in text_lower:
                    detected_skills.add(skill)
        
        # Second pass: word boundary matching (for single-word skills)
        import re
        for skill in skills_database:
            if ' ' not in skill:  # Single-word skills
                # Use word boundaries to avoid partial matches
                # But be flexible with common variations
                pattern = r'\b' + re.escape(skill) + r'\b'
                if re.search(pattern, text_lower):
                    detected_skills.add(skill)
        
        return detected_skills
    
    def extract_keywords(self, text: str, top_n: int = 30) -> Dict[str, List[str]]:
        """
        Extract keywords using multiple methods
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return per method
            
        Returns:
            Dictionary with keywords from different methods
        """
        result = {
            'technical_skills': list(self.extract_technical_skills(text)),
            'tfidf_keywords': self.extract_keywords_tfidf(text, top_n),
        }
        
        if self.use_spacy and self.nlp:
            result['spacy_keywords'] = self.extract_keywords_spacy(text, top_n)
        
        # Combine all keywords
        all_keywords = set()
        for keywords in result.values():
            all_keywords.update(keywords)
        
        result['all_keywords'] = list(all_keywords)
        
        return result
    
    def extract_cv_entities(self, text: str) -> Dict:
        """
        Extract structured CV information using enhanced NLP
        Similar to en_cv_info_extr functionality
        
        Args:
            text: Resume text
            
        Returns:
            Dictionary with extracted CV entities
        """
        if not self.nlp:
            return {}
        
        doc = self.nlp(text)
        
        # Extract entities
        entities = {
            'skills': [],
            'education': [],
            'experience_years': [],
            'organizations': [],
            'locations': [],
            'certifications': [],
            'tools': [],
            'languages': [],
            'dates': []
        }
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ == 'ORG':
                entities['organizations'].append(ent.text)
            elif ent.label_ == 'GPE' or ent.label_ == 'LOC':
                entities['locations'].append(ent.text)
            elif ent.label_ == 'DATE':
                entities['dates'].append(ent.text)
        
        # Extract technical skills using our comprehensive database
        entities['skills'] = list(self.extract_technical_skills(text))
        
        # Extract education keywords
        education_keywords = [
            'bachelor', 'master', 'phd', 'doctorate', 'associate', 'diploma',
            'degree', 'university', 'college', 'institute', 'school',
            'bs', 'ba', 'ms', 'ma', 'mba', 'bsc', 'msc', 'beng', 'meng'
        ]
        text_lower = text.lower()
        for keyword in education_keywords:
            if keyword in text_lower:
                # Find context around the keyword
                import re
                pattern = rf'([^\n.]*{keyword}[^\n.]*)'
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                entities['education'].extend(matches[:3])  # Limit to top 3
        
        # Extract certification keywords
        cert_keywords = [
            'certified', 'certification', 'certificate', 'licensed', 'license',
            'aws', 'azure', 'google cloud', 'pmp', 'cissp', 'comptia', 'itil',
            'scrum master', 'csm', 'cpa', 'cfa', 'phr', 'sphr'
        ]
        for keyword in cert_keywords:
            if keyword in text_lower:
                pattern = rf'([^\n.]*{keyword}[^\n.]*)'
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                entities['certifications'].extend(matches[:3])
        
        # Extract years of experience
        # Look for patterns like "5 years", "5+ years", "5-7 years"
        exp_pattern = r'(\d+)[\+\-\s]*(?:to|-)?\s*(\d+)?\s*(?:years?|yrs?)'
        exp_matches = re.findall(exp_pattern, text_lower)
        for match in exp_matches:
            if match[0]:
                entities['experience_years'].append(match[0])
        
        # Remove duplicates
        for key in entities:
            if isinstance(entities[key], list):
                entities[key] = list(set(entities[key]))
        
        return entities
