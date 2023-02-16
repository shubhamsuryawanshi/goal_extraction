## Natural Language Processing for ESG Goal Extraction

The objective of the project is to augment ESG research by extracting and tracking commitments made my the companies in their annual ESG reports

I am proposing an NLP pipeline which would extract organizational goals from the piece of text provided by the user.

The core stages of the proposed pipeline:
1. Coreference Resolution
2. Named Entity Recognition
3. Parts-of-speech Tagging
4. Fine-tuned BERT Classifier

Generally, goals statements constitute the pattern wherein the organization(company) is an active actor with a potential futuristic timeline. The first three stages help to filter action statements large text corpus and finally the BERT classifier extracts goals from action statements.

The initial 3 stages used for NLP preprocessing for ESG goal extraction would use 2 open-sourced pretrained transformer model-based pipelines.

* First stage – Coreference pipeline  
This experimental pipeline encompasses two prime components:   
a. CoreferenceResolver – An LSTM over the RoBERTa base transfomer model.  
b. SpanResolver – A lightweight CNN with two output channels. 

* Second and Third stage – English core transformer pipeline  
Named Entity Recognition and POS tagging

* Fourth stage – Fine-tuned BERT model  
An annotated dataset was created from ESG report from multiple DAX100 companies to fine-tune a pre-trained model.
