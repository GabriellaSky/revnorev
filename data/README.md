## extension.csv 
Contains samples of argumentative claims, which have been introduced before the reported date of data collection of the original ClaimRev corpus (Jun 26 2020).
Claims that have undergone a revision within 6 months after the initial data collection (Dec 22 2020) have been filtered out.

## combined_data.csv
Combines the original ClaimRev corpus with the extension and includes the train/dev/test split used in all experiments.

### Structure: 
- claim_id  - ids of the claim version
- revision_id - ordinal number of the revision from the revision history 
- claim_text - text of the argumentative claim
- parent_claim - text of the preceding claim that the claim in question is supporting or opposing
- thesis - main thesis of the debate
- label - indicates whether the claim needs further revision or can be considered optimal (see Section 4 for details)
- data_split - denotes whether the samples is used in the train/dev or test set
- 20 additional columns indicating the topical categories each claim belongs to	
	