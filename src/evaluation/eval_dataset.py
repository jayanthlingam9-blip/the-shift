"""RAGAS evaluation dataset.

Hand-authored question / ground_truth pairs spanning the finance, healthcare,
and legal domains, plus out-of-scope questions the pipeline should refuse. Each
item carries a `domain` tag for slicing scores by domain. Consumed by
evaluation.ragas_evaluator.run_ragas_evaluation.
"""

EVAL_DATASET: list[dict[str, str]] = [
    # =========================================================================
    # FINANCIAL DOMAIN — Document 1: JPMorgan Chase Proxy / Annual Report
    # =========================================================================
    {
        "question": "What is the Return on Tangible Common Equity (ROTCE) of JPMorgan Chase, and how does the ROE of its Asset & Wealth Management segment compare to the firm-wide ROE?",
        "ground_truth": "JPMorgan Chase's ROTCE is 20%. The Asset & Wealth Management (AWM) segment has an ROE of 40%, which is significantly higher than the firm-wide ROE of 17%.",
        "domain": "finance",
    },
    {
        "question": "How many of the 11 nominated directors on JPMorgan Chase's board are independent, and which director serves as the Chair of the Risk committee?",
        "ground_truth": "10 out of the 11 nominated directors are independent — all except the CEO, James Dimon. Linda B. Bammann serves as the Chair of the Risk committee.",
        "domain": "finance",
    },
    {
        "question": "What was James Dimon's 2025 annual compensation, and how does JPMorgan Chase's 3-year average annual CEO pay as a percentage of profits compare to its closest peer?",
        "ground_truth": "James Dimon's 2025 annual compensation was $43 million. JPMorgan Chase's 3-year average annual CEO pay as a percentage of profits (2023-2025) is 0.07%, which is the lowest among all listed peers. The closest peer is Bank of America at 0.18%, meaning JPMorgan's ratio is roughly 2.5 times more efficient.",
        "domain": "finance",
    },
    # =========================================================================
    # FINANCIAL DOMAIN — Document 2: Reliance Industries Annual Report 2024-25
    # =========================================================================
    {
        "question": "Which business segment of Reliance Industries has the highest revenue, and what is the total value added by the company in FY 2024-25 compared to FY 2023-24?",
        "ground_truth": "The Oil to Chemicals segment has the highest revenue at Rs 6,26,921 Crore (US$ 73.4 Billion). The total value added in FY 2024-25 is Rs 4,30,453 Crore, up from Rs 3,94,020 Crore in FY 2023-24.",
        "domain": "finance",
    },
    {
        "question": "How many 5G users does Jio have as of March 2025, what percentage of its wireless data traffic do they contribute, and what is Jio's total spectrum footprint after the June 2024 auction?",
        "ground_truth": "As of March 2025, Jio has approximately 191 million 5G users, contributing around 45% of its wireless data traffic. After enhancing its holdings in the 1800 MHz band in Bihar and West Bengal through the June 2024 Spectrum Auction, Jio's total spectrum footprint increased to 26,801 MHz (uplink + downlink).",
        "domain": "finance",
    },
    {
        "question": "What are Reliance Industries' total assets and total equity as at 31st March 2025, and how much did total assets grow compared to 31st March 2024?",
        "ground_truth": "As at 31st March 2025, Reliance Industries' total assets are Rs 10,22,401 Crore and total equity is Rs 5,43,087 Crore. Total assets grew by Rs 62,758 Crore compared to 31st March 2024, when they were Rs 9,59,643 Crore.",
        "domain": "finance",
    },
    # =========================================================================
    # FINANCIAL DOMAIN — Document 3: msft_2025_10k
    # =========================================================================
    {
        "question": "What was the Operating Income for Microsoft's Intelligent Cloud segment in fiscal year 2025, and how does it compare to fiscal year 2023?",
        "ground_truth": "Microsoft's Intelligent Cloud segment reported an Operating Income of $44,589 million in fiscal year 2025, compared to $28,411 million in fiscal year 2023.",
        "domain": "finance",
    },
    {
        "question": "What was Microsoft's net deferred income tax assets as of June 30, 2025, and how was it reported in the balance sheet?",
        "ground_truth": "Microsoft's net deferred income tax assets as of June 30, 2025 was $26,273 million, reported as $29,108 million in Other long-term assets offset by $(2,835) million in Long-term deferred income tax liabilities.",
        "domain": "finance",
    },
    {
        "question": "What was the total number of shares purchased by Microsoft during the fourth quarter of fiscal year 2025, and what was the approximate dollar value of shares that may yet be purchased under the plans or programs as of June 30, 2025?",
        "ground_truth": "Microsoft purchased a total of 7,520,493 shares during the fourth quarter of fiscal year 2025 (April–June 2025). As of June 30, 2025, the approximate dollar value of shares that may yet be purchased under the plans or programs was $57,349 million.",
        "domain": "finance",
    },
    # =========================================================================
    # HEALTHCARE DOMAIN — Document 4: Dailymed ozempic prescribing label
    # =========================================================================
    {
        "question": "In the 24-hour plasma glucose profile study, how many patients were in the OZEMPIC 1 mg end-of-treatment group compared to the OZEMPIC baseline group?",
        "ground_truth": "The OZEMPIC 1 mg end-of-treatment group had 36 patients (n=36), while the OZEMPIC baseline group had 37 patients (n=37).",
        "domain": "healthcare",
    },
    {
        "question": "In the Week 30 OZEMPIC monotherapy trial, what percentage of patients achieved HbA1c less than 7% in the OZEMPIC 0.5 mg group compared to the Placebo group?",
        "ground_truth": "73% of patients in the OZEMPIC 0.5 mg group achieved HbA1c less than 7%, compared to 28% in the Placebo group.",
        "domain": "healthcare",
    },
    {
        "question": "In the Mean HbA1c (%) Over Time study from Baseline to Week 56, how many patients were in each treatment group at randomization and how many remained at Week 30?",
        "ground_truth": "At randomization, OZEMPIC 0.5 mg had 409 patients, OZEMPIC 1 mg had 409 patients, and Sitagliptin had 407 patients. At Week 30, OZEMPIC 0.5 mg had 383 patients, OZEMPIC 1 mg had 378 patients, and Sitagliptin had 387 patients.",
        "domain": "healthcare",
    },
    # =========================================================================
    # HEALTHCARE DOMAIN — Document 5: hcahps hospital records (CSV)
    # Removed from the eval set for now: the 3 HCAHPS CSV questions were
    # dropped to shorten the run while we iterate on metric fixes.
    # =========================================================================
    # =========================================================================
    # HEALTHCARE DOMAIN — Document 6: UnitedHealthcare HEDIS Measures
    # =========================================================================
    {
        "question": "According to UnitedHealthcare's HEDIS measures for Prenatal and Postpartum Care, when must a prenatal care visit take place, and why does a Pap test alone not qualify as a prenatal care visit?",
        "ground_truth": "A prenatal care visit must take place in the first trimester, on or before the enrollment start date, or within 42 days of enrollment with the health plan. The first trimester is defined as 280-176 days prior to delivery/EDD. A Pap test does not count as a prenatal care visit, and a colposcopy alone does not meet numerator compliance for prenatal care. The visit must be with an OB-GYN or prenatal care provider and must include at least one qualifying test or procedure such as a diagnosis of pregnancy, auscultation for fetal heart tone, documentation in a standard prenatal flowsheet, or prenatal lab results.",
        "domain": "healthcare",
    },
    {
        "question": "What is the definition of the Controlling High Blood Pressure (CBP) HEDIS measure, and what are the specific CPT II codes used to report systolic blood pressure levels below 130 mmHg versus 140 mmHg or above?",
        "ground_truth": "The CBP HEDIS measure tracks the percentage of members ages 18-85 who had a diagnosis of hypertension (HTN) and whose blood pressure was adequately controlled at <140/90 mmHg during the measurement year. The CPT II code for systolic blood pressure level <130 mmHg is 3074F, for levels 130-139 mmHg is 3075F, and for levels >=140 mmHg is 3077F. The measure applies to Commercial, Exchange/Marketplace, Medicaid, and Medicare plans, and uses a hybrid collection method including claim/encounter data, medical record documentation, and pharmacy data.",
        "domain": "healthcare",
    },
    {
        "question": "What are the ICD-10 diagnosis codes for diabetes mellitus without complications used in the UnitedHealthcare Eye Exam for Diabetes (EED) HEDIS measure, and what LOINC codes differentiate left eye versus right eye retinal exam findings?",
        "ground_truth": "The ICD-10 diagnosis codes for diabetes mellitus without complications are E10.9, E11.9, and E13.9. For retinal exam findings, the left eye uses LOINC code 71490-7 and the right eye uses LOINC code 71491-5. Diabetic retinopathy severity level is captured using LOINC codes LA18644-7, LA18645-4, LA18643-9, LA18648-8, and LA18646-2. A finding of no retinopathy uses LOINC code LA18643-9 specifically.",
        "domain": "healthcare",
    },
    # =========================================================================
    # LEGAL DOMAIN — Document 7: GDPR
    # =========================================================================
    {
        "question": "According to the GDPR, what specific categories of personal data processing are identified as posing risks to the rights and freedoms of natural persons, and what types of damage can such processing lead to?",
        "ground_truth": "The GDPR identifies that processing posing risks includes: processing that reveals racial or ethnic origin, political opinions, religion or philosophical beliefs, trade union membership; processing of genetic data, health data, data concerning sex life, or criminal convictions and offences. It also includes evaluating personal aspects such as performance at work, economic situation, health, personal preferences, reliability or behaviour, location or movements to create or use personal profiles; processing data of vulnerable persons, particularly children; and processing involving large amounts of data affecting a large number of data subjects. The types of damage include physical, material, or non-material damage, specifically discrimination, identity theft or fraud, financial loss, damage to reputation, loss of confidentiality of professionally protected data, unauthorised reversal of pseudonymisation, or any other significant economic or social disadvantage.",
        "domain": "legal",
    },
    {
        "question": "Under GDPR Recital 81, what guarantees must a controller look for when selecting a data processor, and what must happen to personal data after the processing is completed on behalf of the controller?",
        "ground_truth": "When selecting a data processor, the controller must use only processors providing sufficient guarantees in terms of expert knowledge, reliability, and resources to implement technical and organisational measures that meet the requirements of the GDPR, including security of processing. The processor's adherence to an approved code of conduct or an approved certification mechanism may be used to demonstrate compliance. After the completion of processing on behalf of the controller, the processor must, at the choice of the controller, return or delete the personal data, unless there is a requirement to store the personal data under Union or Member State law. The processing relationship must be governed by a contract or legal act setting out the subject-matter, duration, nature and purposes of processing, type of personal data, and categories of data subjects.",
        "domain": "legal",
    },
    {
        "question": "Under GDPR Article 28, what happens if a data processor engages a sub-processor who fails to fulfil its data protection obligations, and under what circumstance does a processor become considered a controller?",
        "ground_truth": "When a processor engages another sub-processor, the same data protection obligations from the contract between the controller and the original processor must be imposed on the sub-processor. If that sub-processor fails to fulfil its data protection obligations, the initial processor remains fully liable to the controller for the performance of that sub-processor's obligations. Additionally, the sub-processor can only be engaged with prior specific or general written authorisation from the controller. A processor becomes considered a controller if it infringes the Regulation by independently determining the purposes and means of processing, as stated in Article 28(10).",
        "domain": "legal",
    },
    # =========================================================================
    # LEGAL DOMAIN — Document 8: RBI Basel III Capital Regulations
    # =========================================================================
    {
        "question": "According to the RBI Basel III Capital Regulations, how is surplus Total Capital of a subsidiary calculated for the purpose of determining minority interest recognition, and what is the minimum Total Capital requirement percentage including the capital conservation buffer?",
        "ground_truth": "Surplus Total Capital of a subsidiary is calculated as the Total Capital of the subsidiary minus the lower of: (a) the minimum Total Capital requirement of the subsidiary plus the capital conservation buffer, which equals 11.5% of risk-weighted assets, or (b) the portion of the consolidated minimum Total Capital requirement plus the capital conservation buffer (11.5% of consolidated risk-weighted assets) that relates to the subsidiary. The amount of surplus Total Capital attributable to third party investors is then calculated by multiplying the surplus Total Capital by the percentage of Total Capital held by third party investors.",
        "domain": "legal",
    },
    {
        "question": "Under the RBI Basel III Capital Regulations, what are the different treatment approaches for Deferred Tax Assets (DTAs) associated with accumulated losses versus those related to timing differences, and what is the combined cap for limited recognition of DTAs and significant investments in unconsolidated financial entities?",
        "ground_truth": "DTAs associated with accumulated losses must be deducted in full from CET1 capital. DTAs related to timing differences (other than accumulated losses) may be recognized in CET1 capital up to 10% of the bank's CET1 capital instead of full deduction. However, this limited recognition of timing-difference DTAs, combined with limited recognition of significant investments in common shares of unconsolidated financial entities, must not exceed 15% of CET1 capital (calculated after all regulatory adjustments). Banks must also ensure that CET1 capital after applying the 15% combined limit does not result in recognizing any individual item beyond its 10% individual limit. DTAs not deducted from CET1 capital are risk-weighted at 250%. DTAs may be netted with associated deferred tax liabilities (DTLs) only if both relate to the same taxation authority and offsetting is permitted.",
        "domain": "legal",
    },
    {
        "question": "How does the RBI Basel III framework treat the cash flow hedge reserve in CET1 capital calculation, and why does it require derecognition of the reserve for hedging items not fair valued on the balance sheet?",
        "ground_truth": "The RBI Basel III framework requires that the cash flow hedge reserve relating to hedging of items not fair valued on the balance sheet (including projected cash flows) be derecognised from CET1 capital. Positive amounts must be deducted and negative amounts must be added back. The rationale is that it removes artificial volatility in Common Equity because the reserve only reflects one half of the picture: the fair value of the derivative, but not the changes in fair value of the hedged future cash flow. At the consolidated level, this derecognition applies to the cash flow hedge reserve attributed to subsidiaries in addition to the derecognition pertaining to the solo bank.",
        "domain": "legal",
    },
    # =========================================================================
    # LEGAL DOMAIN — Document 9: SEBI Master Circular (ICDR)
    # =========================================================================
    {
        "question": "According to the SEBI Master Circular, where must Merchant Bankers file offer documents online, and what is the specific filing process for rights issues that differs from the standard filing procedure?",
        "ground_truth": "Merchant Bankers must file offer documents and related documents online through the SEBI Intermediary Portal at https://siportal.sebi.gov.in, simultaneously with physical filing. However, for rights issues, the process differs — the issuer must file the letter of offer with SEBI through email at cfddil@sebi.gov.in, and the payment of filing fees must be made online through a payment link provided on the SEBI website under the fees category Filing Fees. Draft offer documents that are not compliant with Schedule VI of the ICDR Regulations may be returned to the issuer.",
        "domain": "legal",
    },
    {
        "question": "What are the SEBI requirements for the Audiovisual (AV) presentation of public issue disclosures, including duration, language format, and the specific warning that must be included about finfluencers?",
        "ground_truth": "SEBI requires that AV presentations be prepared for all main board public issues covering salient disclosures from the DRHP, RHP, and Price Band Advertisement. Each bilingual version must be approximately 10 minutes in duration, initially in English and Hindi, with the Hindi version containing text in Devanagari script. The AV must include a specific disclosure warning investors not to rely on any document, content, or information provided on the internet, websites, social media platforms, or micro-blogging platforms by finfluencers, and to rely only on the Offer document and Price Band Advertisement for investment decisions. The content must be factual, non-repetitive, non-promotional, and not misleading. The AV must be uploaded on the website of the Issuer and the Association of Investment Bankers of India (AIBI) within 5 working days of filing the DRHP.",
        "domain": "legal",
    },
    {
        "question": "What is the SEBI-mandated formula for calculating minimum fair compensation to retail investors in an IPO when an SCSB fails to process their application, and what happens if the listing price is below the issue price?",
        "ground_truth": "The SEBI-mandated formula is: Compensation = (Listing price - Issue Price) x Number of shares that would have been allotted if bid was successful x Probability of allotment of shares determined on the basis of allotment. The listing price is taken as the highest of the opening prices on the day of listing across recognized stock exchanges. No compensation is payable if the listing price is below the issue price. For issues subscribed between 90-100% (non-oversubscribed), applicants are compensated for all shares they would have been allotted. Investors must seek redressal within three months of the listing date, and the SCSB must resolve complaints within 15 days, failing which it must pay interest at 15% per annum for any delay beyond that period.",
        "domain": "legal",
    },
    # =========================================================================
    # OUT OF SCOPE — Should refuse to answer
    # =========================================================================
    {
        "question": "What is the weather today?",
        "ground_truth": "This question is outside the scope of the provided documents.",
        "domain": "out_of_scope",
    },
    {
        "question": "Who is the president of the United States?",
        "ground_truth": "This question is outside the scope of the provided documents.",
        "domain": "out_of_scope",
    },
    {
        "question": "What is the capital of India?",
        "ground_truth": "This question is outside the scope of the provided documents.",
        "domain": "out_of_scope",
    },
    {
        "question": "What is Reliance's policy on space exploration?",
        "ground_truth": "The provided documents do not contain information about Reliance's space exploration policy.",
        "domain": "out_of_scope",
    },
]
