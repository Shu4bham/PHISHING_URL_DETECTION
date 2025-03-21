# Phishing_Detection



This appears to be a phishing detection project that uses machine learning to identify potentially malicious URLs and domains. Let me break down the main components:

1. **Core Machine Learning (algorithm.py)**:
- Implements two ML models: Naive Bayes and Random Forest Classifier
- Handles training data preparation and model creation
- Processes test data and generates predictions
- Includes functionality for cross-validation and confusion matrix generation

2. **Domain Processing (domain_parser.py)**:
- Parses URLs into components (domain, subdomain, TLD, path, etc.)
- Extracts raw words from different URL components
- Handles both labeled and unlabeled samples

3. **Feature Extraction (url_rules.py, active_rules.py)**:
- Extracts various URL-based features like:
  - Length metrics
  - Digit counts
  - Special character presence
  - TLD validation
  - Alexa rank checking
  - Brand name detection
  - Punnycode detection
  - Character repetition patterns

4. **NLP Components (word_with_nlp.py, word_splitter_file.py)**:
- Performs natural language processing on URL components
- Splits compound words
- Detects brand names and keywords
- Identifies random/DGA-like strings
- Us
es edit distance to find similar words

5. **Google Safe Browsing Integration (active_rules.py)**:
- Integrates with Google Safe Browsing API
- Checks URLs against known phishing/malware databases

6. **Support Modules**:
- `json2arff.py`: Converts between JSON and ARFF formats (for WEKA compatibility)
- `ns_log.py`: Logging functionality
- `gib_detect.py`: Gibberish detection for random string identification

The system appears to work by:
1. Taking URLs as input
2. Parsing them into components
3. Extracting various features (URL-based, linguistic, reputation-based)
4. Running them through ML models
5. Generating predictions about whether URLs are legitimate or phishing attempts
#   P H I S H I N G _ U R L _ D E T E C T I O N  
 