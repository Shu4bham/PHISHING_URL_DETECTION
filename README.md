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




Let's count the number of features extracted from a raw URL based on the `url_rules.py` file you provided. We'll go through the `rules_main` function and each of the rule functions it calls to determine how many features are generated.

Here's a breakdown of the features from each function in `url_rules.py`'s `rules_main`:

1.  **`digit_count(domain, subdomain, path)`:**
    *   `domain_digit_count`: 1 feature
    *   `subdomain_digit_count`: 1 feature
    *   `path_digit_count`: 1 feature
    *   **Total: 3 features**

2.  **`length(domain, subdomain, path)`:**
    *   `domain_length`: 1 feature
    *   `subdomain_length`: 1 feature
    *   `path_length`: 1 feature
    *   **Total: 3 features**

3.  **`tld_check(tld)`:**
    *   `isKnownTld`: 1 feature
    *   **Total: 1 feature**

4.  **`check_rule_5(words_raw)` (Rule 5 - "www" and "com" checks):**
    *   `www`: 1 feature (count of "www" in words)
    *   `com`: 1 feature (count of "com" in words)
    *   **Total: 2 features**

5.  **`punny_code(domain)`:**
    *   `punnyCode`: 1 feature (binary, 1 if punycode, 0 otherwise)
    *   **Total: 1 feature**

6.  **`random_domain(domain)`:**
    *   `random_domain`: 1 feature (binary, 1 if random-like, 0 otherwise - based on gibberish model)
    *   **Total: 1 feature**

7.  **`subdomain_count(subdomain)`:**
    *   `subDomainCount`: 1 feature (count of subdomains)
    *   **Total: 1 feature**

8.  **`char_repeat(words_raw)`:**
    *   `char_repeat`: 1 feature (count of character repetitions of length 2, 3, 4, or 5)
    *   **Total: 1 feature**

9.  **`alexa_check(domain, tld)`:**
    *   `alexa1m_tld`: 1 feature (binary, 1 if domain+tld in Alexa Top 1M list, 0 otherwise - TLD included)
    *   `alexa1m`: 1 feature (binary, 1 if domain in Alexa Top 1M list, 0 otherwise - TLD excluded)
    *   **Total: 2 features**

10. **`special_chars(domain, subdomain, path)`:**
    *   `-`: 1 feature (count of hyphens)
    *   `.`: 1 feature (count of dots)
    *   `/`: 1 feature (count of slashes)
    *   `@`: 1 feature (count of `@` symbols)
    *   `?`: 1 feature (count of `?` symbols)
    *   `&`: 1 feature (count of `&` symbols)
    *   `=`: 1 feature (count of `=` symbols)
    *   `_`: 1 feature (count of underscores)
    *   **Total: 8 features**

11. **`check_domain_in_list(domain)`:**
    *   `domain_in_brand_list`: 1 feature (binary, 1 if domain is in brand list, 0 otherwise)
    *   **Total: 1 feature**

12. **`nlp_features(words_raw)` (NLP-based features - *these are multiple features*):**
    *   This function itself calls `self.nlp_manager.evaluate(...)` which returns a dictionary with both `info` and `features`. We are interested in the `features` part. Let's look at the `nlp_class.py` and `nlp_class.evaluate()` function to count NLP features.

    Looking at `word_with_nlp.py` and the `nlp_class.evaluate()` function, the `result['features']` dictionary in `nlp_features` contains:

    *   `raw_word_count`: 1 feature
    *   `splitted_word_count`: 1 feature
    *   `average_word_length`: 1 feature
    *   `longest_word_length`: 1 feature
    *   `shortest_word_length`: 1 feature
    *   `std_word_length`: 1 feature
    *   `compound_word_count`: 1 feature
    *   `keyword_count`: 1 feature
    *   `brand_name_count`: 1 feature
    *   `negligible_word_count`: 1 feature
    *   `target_brand_count`: 1 feature
    *   `target_keyword_count`: 1 feature
    *   `similar_keyword_count`: 1 feature
    *   `similar_brand_count`: 1 feature
    *   `average_compound_words`: 1 feature
    *   `random_words`: 1 feature
    *   **Total NLP Features: 16 features**

**Total Count of URL-Based and NLP Features:**

Adding up the features from all rule functions:

3 + 3 + 1 + 2 + 1 + 1 + 1 + 1 + 2 + 8 + 1 + 16 = **40 features**

**Therefore, based on the `url_rules.py` and `word_with_nlp.py` code, your system, in its current configuration, extracts approximately 40 features from a raw URL.**

**Important Notes:**

*   **NLP Features are a Significant Portion:**  NLP-based features (`nlp_features`) contribute a substantial portion (16 out of 40) of the total features, highlighting the importance of natural language processing in your approach.
*   **Active Rules (Currently Not Counted):**  The `active_rules.goog_safe_browsing()` function (which is commented out in `rule_extraction.py`) would add *additional* features if it were fully implemented and called.  Currently, as it's not active in the main pipeline, we haven't counted features from `active_rules.py` in this total. If you were to uncomment and activate the Google Safe Browsing check, it would add at least one more feature (`google_safe_browsing`).
*   **Feature List in Paper (Table 2):**  The research paper extract you provided mentions "Table 2 NLP based features".  If you have access to the full "Table 2" from the paper "Expert Systems With Applications 117 (2019) 345-357", comparing your counted features with the features listed in Table 2 of the paper would be a very valuable step to ensure your implementation aligns with the research.  Table 2 might also give you more details about the specific NLP features they used.
*   **Dynamic Feature Count:** The exact number of features might slightly vary depending on the input URL and how certain rules apply. For example, if a URL doesn't have a subdomain, the `subdomain_length` and `subdomain_digit_count` might be zero, but they are still counted as features in the feature vector.

This count of approximately **40 features** provides a good understanding of the feature space your phishing detection system is currently using, based on the provided code.



**1. `word_with_nlp.py` - Core NLP Functionality:**

*   **Purpose:** This file is dedicated to implementing NLP-related functionalities for analyzing words extracted from URLs.
*   **`nlp_class` Class:** This class encapsulates the NLP logic. Key functionalities include:
    *   **Keyword and Brand Name Recognition:**  It uses lists of keywords (`keywords.txt`) and brand names (`allbrand.txt`) to identify if words in the URL are known keywords or brand names.
    *   **Similarity to Keywords and Brands:** It uses `editdistance` to calculate the edit distance (Levenshtein distance) between words in the URL and known keywords/brands. This helps detect words that are *similar* to keywords or brands (typosquatting, slight variations).
    *   **Gibberish (Random Word) Detection:** It loads the `gib_model.pki` model (trained by `gib_detect_train.py`) and uses it to determine if words in the URL are random-looking or like gibberish. This is crucial for detecting randomly generated domain names.
    *   **Word Categorization and Grouping:** The `parse()` method groups words from the URL into categories like:
        *   `keywords_in_words`: Words that are directly from the keyword list.
        *   `brands_in_words`: Words that are direct brand names.
        *   `similar_to_brands`: Words similar to brand names.
        *   `similar_to_keywords`: Words similar to keywords.
        *   `dga_in_words`: Words detected as Domain Generation Algorithm (DGA) words (gibberish/random).
        *   `len_lt_7`: Words with length less than 7 characters.
        *   `len_gt_7`: Words with length greater than 7 characters (potential compound words).
    *   **Fraud Analysis:** The `fraud_analysis()` method further analyzes the words, identifying `found_keywords`, `found_brands`, `similar_to_keywords`, `similar_to_brands`, and `other_words`.
    *   **Feature Evaluation and Statistics:** The `evaluate()` method calculates various statistical features based on the categorized words, such as:
        *   Counts of keywords, brands, random words, compound words, etc.
        *   Average, longest, shortest word lengths.
        *   Standard deviation of word lengths.

**2. `url_rules.py` - `nlp_features()` Function:**

*   **Integration Point:** The `url_rules.py` file, specifically the `nlp_features()` function, is where the `nlp_class` is used to integrate NLP features into the overall URL feature set.
*   **Calling `nlp_class`:** The `nlp_features()` function in `url_rules.py` does the following:
    *   Instantiates `nlp_class` (`self.nlp_manager = nlp_class()` in `url_rules.__init__`).
    *   Calls `self.nlp_manager.parse(words_raw)` to categorize words from the URL.
    *   Calls `self.word_splitter.splitl(...)` to split compound words (words longer than 7 characters).
    *   Calls `self.nlp_manager.fraud_analysis(...)` to perform more in-depth analysis of the words.
    *   Calls `self.nlp_manager.evaluate(...)` to calculate the final set of NLP-based features.
*   **NLP Features in Feature Set:** The `nlp_features()` function returns a dictionary of NLP-based features, which are then included in the overall feature set for each URL.

**3. Types of NLP Techniques Used:**

*   **Lexicon-Based Analysis:** Using keyword lists (`keywords.txt`) and brand name lists (`allbrand.txt`) is a lexicon-based approach.
*   **Edit Distance (Levenshtein Distance):**  Used for fuzzy matching and detecting words that are similar to known keywords or brands (useful for typosquatting detection).
*   **Statistical Language Modeling (N-gram based with Markov Chains):** The `gib_detect_train.py` and `gib_model.pki` implement a simple statistical language model to detect gibberish or random strings. This is a form of NLP, although a relatively basic one.
*   **Word Splitting:**  The `WordSplitterClass` attempts to split compound words into meaningful sub-words, which is a basic form of word segmentation, relevant to NLP.
*   **Statistical Features from Text:**  Calculating word counts, word lengths, and their statistics are also considered basic NLP-related feature engineering.

**Purpose of NLP in Phishing Detection (in this project):**

The NLP techniques are used in this project to:

*   **Understand the Textual Content of URLs:** URLs are not just random strings of characters; they often contain words that have meaning. NLP helps to extract this meaning.
*   **Identify Keywords and Brands in URLs:** Phishing URLs often try to incorporate keywords related to sensitive actions (login, secure, account) or brand names of targeted organizations to appear legitimate. NLP helps detect these.
*   **Detect Typosquatting and Brand Impersonation:** By using edit distance, NLP helps identify URLs that use slightly modified or misspelled brand names to trick users.
*   **Recognize Random or Gibberish Domain Names:**  NLP-based gibberish detection helps identify URLs that use randomly generated domain names, which are often associated with malicious activities.
*   **Extract Higher-Level Features:**  NLP allows you to go beyond simple character-based features and create more semantic or word-level features that can be more informative for phishing detection.

**In summary, NLP is a core part of  project's feature engineering strategy. It allows to analyze the textual components of URLs in a more sophisticated way, going beyond simple string matching and character counts, and enabling the detection of patterns and features that are indicative of phishing attempts.** The NLP components are primarily implemented in `word_with_nlp.py` and integrated into the URL feature extraction process within `url_rules.py` and `rule_extraction.py`.